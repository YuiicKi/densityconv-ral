/**
 * Batch evaluation on RDK S100 — end-to-end C++ inference for nuScenes val set.
 * Works for both DensityConv (2 BPU inputs: bev+mask) and Conv2d-only (1 input: bev).
 *
 * Build: g++ -O3 -march=armv8-a -funroll-loops -o eval_batch_inst eval_batch_inst.cpp \
 *        -I/usr/include/hobot -L/usr/hobot/lib -ldnn -lhbrt4 -lhbucp -lpthread
 *
 * Usage: ./eval_batch_inst <model.hbm> <vfe_dir> <index.txt> <data_root> <output.json> [max_frames] [serial]
 *   serial=1: fully sequential per-frame (load→VFE→BPU→decode/NMS), logs true single-frame latency
 * Timing: writes <output.json>.timing.csv (per-frame stage breakdown), prints VmHWM at end
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <string>

#include <arm_neon.h>

#include "dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

#define CHECK(call) do { int32_t r = (call); if (r) { fprintf(stderr, "ERR %s=%d @%d\n", #call, r, __LINE__); exit(1); } } while(0)

#ifndef VFE_THREADS
#define VFE_THREADS 6
#endif

// ============================================================
// Constants
// ============================================================
static const float PC_RANGE[6] = {-54.4f, -54.4f, -5.0f, 54.4f, 54.4f, 3.0f};
static const float VOXEL_SIZE[2] = {0.1f, 0.1f};
static const int GRID_W = 1088, GRID_H = 1088, BEV_C = 32;
static const int FEAT_H = 136, FEAT_W = 136;
static const float SCORE_THRESH = 0.1f;
static const float NMS_IOU_THRESH = 0.2f;
static const int TOP_K = 83;
static const float FEAT_STRIDE = 8.0f;
static const int CLASSES_PER_HEAD[] = {1, 2, 2, 1, 2, 2};
static const int HEAD_CLS_OFFSET[] = {0, 1, 3, 5, 6, 8};
static const int MAX_DETS_PER_SAMPLE = 500;

static const char* DET_NAMES[] = {
    "car","truck","construction_vehicle","bus","trailer",
    "barrier","motorcycle","bicycle","pedestrian","traffic_cone"
};

// ============================================================
// Timer
// ============================================================
struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double ms() { return std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-t0).count(); }
};

// Peak RSS (kernel high-water mark), kB; -1 on failure
static long read_vmhwm_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256]; long v = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) { sscanf(line + 6, "%ld", &v); break; }
    }
    fclose(f);
    return v;
}

// ============================================================
// Load .pcd.bin (N x 5 float32)
// ============================================================
static std::vector<float> load_points(const char* path, int& n_points) {
    FILE* f = fopen(path, "rb");
    if (!f) { n_points = 0; return {}; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    n_points = sz / (5 * sizeof(float));
    std::vector<float> pts(n_points * 5);
    fread(pts.data(), sizeof(float), n_points * 5, f);
    fclose(f);
    return pts;
}

// ============================================================
// VFE weights (NEON SIMD)
// ============================================================
static float fc_w[32][8];
static float bn_w[32], bn_b[32], bn_mean[32], bn_var[32];
static const float BN_EPS = 1e-5f;
static float fc_w_T[8 * 32] __attribute__((aligned(16)));
static float bn_shift_v[32] __attribute__((aligned(16)));
static bool simd_weights_ready = false;

static void prep_simd_weights() {
    for (int c = 0; c < 32; c++) {
        float scale = bn_w[c] / sqrtf(bn_var[c] + BN_EPS);
        bn_shift_v[c] = bn_b[c] - bn_mean[c] * scale;
        for (int k = 0; k < 8; k++)
            fc_w_T[k * 32 + c] = fc_w[c][k] * scale;
    }
    simd_weights_ready = true;
}

static void load_vfe_weights(const char* dir) {
    auto load = [&](const char* name, float* buf, int n) {
        char path[512]; snprintf(path, sizeof(path), "%s/%s", dir, name);
        FILE* f = fopen(path, "rb"); fread(buf, sizeof(float), n, f); fclose(f);
    };
    load("vfe_pfn_layers_0_linear_weight.bin", (float*)fc_w, 32*8);
    load("vfe_pfn_layers_0_norm_weight.bin", bn_w, 32);
    load("vfe_pfn_layers_0_norm_bias.bin", bn_b, 32);
    load("vfe_pfn_layers_0_norm_running_mean.bin", bn_mean, 32);
    load("vfe_pfn_layers_0_norm_running_var.bin", bn_var, 32);
}

// ============================================================
// Real VFE: NEON SIMD, 6-thread, matches OpenPCDet DynamicPillarVFESimple2D
// ============================================================
static void real_vfe(const float* pts, int n_pts, float* bev, float* mask) {
    memset(bev, 0, BEV_C * GRID_H * GRID_W * sizeof(float));
    if (mask) memset(mask, 0, GRID_H * GRID_W * sizeof(float));

    static std::vector<int> cnt;
    static std::vector<float> pillar_feat;
    static bool alloc = false;
    if (!alloc) {
        cnt.resize(GRID_H * GRID_W, 0);
        pillar_feat.resize(GRID_H * GRID_W * 32, 0);
        alloc = true;
    }
    memset(cnt.data(), 0, cnt.size() * sizeof(int));
    memset(pillar_feat.data(), 0, pillar_feat.size() * sizeof(float));

    static std::vector<int> pt_pillar;
    static std::vector<int> pt_idx;
    pt_pillar.clear(); pt_idx.clear();

    const float voxel_z = 0.2f;
    const float x_offset = VOXEL_SIZE[0] / 2 + PC_RANGE[0];
    const float y_offset = VOXEL_SIZE[1] / 2 + PC_RANGE[1];
    const float z_offset = voxel_z / 2 + PC_RANGE[2];

    for (int i = 0; i < n_pts; i++) {
        float x = pts[i*5], y = pts[i*5+1], z = pts[i*5+2];
        if (x < PC_RANGE[0] || x >= PC_RANGE[3] ||
            y < PC_RANGE[1] || y >= PC_RANGE[4] ||
            z < PC_RANGE[2] || z >= PC_RANGE[5]) continue;
        int xi = (int)((x - PC_RANGE[0]) / VOXEL_SIZE[0]);
        int yi = (int)((y - PC_RANGE[1]) / VOXEL_SIZE[1]);
        if (xi < 0 || xi >= GRID_W || yi < 0 || yi >= GRID_H) continue;
        int idx = yi * GRID_W + xi;
        cnt[idx]++;
        pt_pillar.push_back(idx);
        pt_idx.push_back(i);
    }

    if (!simd_weights_ready) prep_simd_weights();

    int n_valid = (int)pt_pillar.size();
    auto worker = [&](int start, int end) {
        const float32x4_t shift0 = vld1q_f32(&bn_shift_v[0]);
        const float32x4_t shift1 = vld1q_f32(&bn_shift_v[4]);
        const float32x4_t shift2 = vld1q_f32(&bn_shift_v[8]);
        const float32x4_t shift3 = vld1q_f32(&bn_shift_v[12]);
        const float32x4_t shift4 = vld1q_f32(&bn_shift_v[16]);
        const float32x4_t shift5 = vld1q_f32(&bn_shift_v[20]);
        const float32x4_t shift6 = vld1q_f32(&bn_shift_v[24]);
        const float32x4_t shift7 = vld1q_f32(&bn_shift_v[28]);
        const float32x4_t zero = vdupq_n_f32(0.0f);

        for (int p = start; p < end; p++) {
            int idx = pt_pillar[p];
            int i = pt_idx[p];
            int yi = idx / GRID_W, xi = idx % GRID_W;
            float x = pts[i*5+0], y = pts[i*5+1], z = pts[i*5+2];

            float feat[8] __attribute__((aligned(16)));
            feat[0] = x - (xi * VOXEL_SIZE[0] + x_offset);
            feat[1] = y - (yi * VOXEL_SIZE[1] + y_offset);
            feat[2] = z - z_offset;
            feat[3] = x; feat[4] = y; feat[5] = z;
            feat[6] = pts[i*5+3]; feat[7] = pts[i*5+4];

            float32x4_t a0=shift0, a1=shift1, a2=shift2, a3=shift3;
            float32x4_t a4=shift4, a5=shift5, a6=shift6, a7=shift7;
            #define FMA_K(k) do { \
                float32x4_t fv = vdupq_n_f32(feat[k]); \
                const float* wp = &fc_w_T[(k)*32]; \
                a0=vfmaq_f32(a0,fv,vld1q_f32(wp+0));  a1=vfmaq_f32(a1,fv,vld1q_f32(wp+4)); \
                a2=vfmaq_f32(a2,fv,vld1q_f32(wp+8));  a3=vfmaq_f32(a3,fv,vld1q_f32(wp+12)); \
                a4=vfmaq_f32(a4,fv,vld1q_f32(wp+16)); a5=vfmaq_f32(a5,fv,vld1q_f32(wp+20)); \
                a6=vfmaq_f32(a6,fv,vld1q_f32(wp+24)); a7=vfmaq_f32(a7,fv,vld1q_f32(wp+28)); \
            } while(0)
            FMA_K(0); FMA_K(1); FMA_K(2); FMA_K(3);
            FMA_K(4); FMA_K(5); FMA_K(6); FMA_K(7);
            #undef FMA_K

            a0=vmaxq_f32(a0,zero); a1=vmaxq_f32(a1,zero);
            a2=vmaxq_f32(a2,zero); a3=vmaxq_f32(a3,zero);
            a4=vmaxq_f32(a4,zero); a5=vmaxq_f32(a5,zero);
            a6=vmaxq_f32(a6,zero); a7=vmaxq_f32(a7,zero);

            float result[32] __attribute__((aligned(16)));
            vst1q_f32(&result[0],a0);  vst1q_f32(&result[4],a1);
            vst1q_f32(&result[8],a2);  vst1q_f32(&result[12],a3);
            vst1q_f32(&result[16],a4); vst1q_f32(&result[20],a5);
            vst1q_f32(&result[24],a6); vst1q_f32(&result[28],a7);

            float* dst = &pillar_feat[idx * 32];
            for (int c = 0; c < 32; c++) {
                const float val = result[c];
                unsigned int desired_bits;
                memcpy(&desired_bits, &val, sizeof(desired_bits));
                auto* addr = reinterpret_cast<unsigned int*>(&dst[c]);
                unsigned int old_bits = __atomic_load_n(addr, __ATOMIC_RELAXED);
                for (;;) {
                    float old_val;
                    memcpy(&old_val, &old_bits, sizeof(old_val));
                    if (!(val > old_val)) break;
                    if (__atomic_compare_exchange_n(
                            addr, &old_bits, desired_bits, false,
                            __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                        break;
                }
            }
        }
    };

    const int NT = VFE_THREADS;
    std::vector<std::thread> threads;
    int chunk = (n_valid + NT - 1) / NT;
    for (int t = 0; t < NT; t++) {
        int s = t*chunk, e = std::min(s+chunk, n_valid);
        if (s < e) threads.emplace_back(worker, s, e);
    }
    for (auto& th : threads) th.join();

    for (int idx = 0; idx < GRID_H * GRID_W; idx++) {
        if (cnt[idx] == 0) continue;
        if (mask) mask[idx] = 1.0f;
        for (int c = 0; c < 32; c++)
            bev[c * GRID_H * GRID_W + idx] = pillar_feat[idx * 32 + c];
    }
}

// ============================================================
// Box3D + Rotated BEV IoU NMS
// ============================================================
struct Box3D {
    float x, y, z, w, l, h, yaw, vx, vy, score;
    int cls;
};

struct Vec2 { float x, y; };

static void get_corners(const Box3D& b, Vec2 c[4]) {
    float cos_y = cosf(b.yaw), sin_y = sinf(b.yaw);
    float hw = b.w/2, hl = b.l/2;
    float dx[4] = {-hw, hw, hw, -hw};
    float dy[4] = {-hl, -hl, hl, hl};
    for (int i = 0; i < 4; i++) {
        c[i].x = b.x + dx[i]*cos_y - dy[i]*sin_y;
        c[i].y = b.y + dx[i]*sin_y + dy[i]*cos_y;
    }
}

static float cross2d(Vec2 o, Vec2 a, Vec2 b) {
    return (a.x-o.x)*(b.y-o.y) - (a.y-o.y)*(b.x-o.x);
}

static float polygon_area(Vec2* poly, int n) {
    if (n < 3) return 0;
    float area = 0;
    for (int i = 0; i < n; i++) {
        int j = (i+1)%n;
        area += poly[i].x*poly[j].y - poly[j].x*poly[i].y;
    }
    return fabsf(area) * 0.5f;
}

static int clip_polygon(Vec2* poly, int n, Vec2 a, Vec2 b, Vec2* out) {
    if (n == 0) return 0;
    int m = 0;
    for (int i = 0; i < n; i++) {
        Vec2 c = poly[i], d = poly[(i+1)%n];
        float cc = cross2d(a, b, c), cd = cross2d(a, b, d);
        if (cc >= 0) out[m++] = c;
        if ((cc > 0) != (cd > 0) && !(cc == 0 || cd == 0)) {
            float t = cc / (cc - cd);
            out[m++] = {c.x + t*(d.x-c.x), c.y + t*(d.y-c.y)};
        }
    }
    return m;
}

static float bev_iou(const Box3D& a, const Box3D& b) {
    Vec2 ca[4], cb[4];
    get_corners(a, ca); get_corners(b, cb);
    Vec2 buf1[16], buf2[16];
    memcpy(buf1, ca, 4*sizeof(Vec2));
    int n = 4;
    for (int i = 0; i < 4 && n > 0; i++) {
        int m = clip_polygon(buf1, n, cb[i], cb[(i+1)%4], buf2);
        memcpy(buf1, buf2, m*sizeof(Vec2));
        n = m;
    }
    float inter = polygon_area(buf1, n);
    float ua = a.w*a.l + b.w*b.l - inter;
    return ua > 0 ? inter/ua : 0;
}

static void nms(std::vector<Box3D>& boxes) {
    std::sort(boxes.begin(), boxes.end(), [](const Box3D& a, const Box3D& b){ return a.score > b.score; });
    std::vector<bool> suppressed(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); i++) {
        if (suppressed[i]) continue;
        for (size_t j = i+1; j < boxes.size(); j++) {
            if (suppressed[j]) continue;
            if (bev_iou(boxes[i], boxes[j]) > NMS_IOU_THRESH)
                suppressed[j] = true;
        }
    }
    std::vector<Box3D> kept;
    for (size_t i = 0; i < boxes.size(); i++)
        if (!suppressed[i]) kept.push_back(boxes[i]);
    boxes = kept;
}

// ============================================================
// CenterPoint decode + per-class NMS → return boxes
// ============================================================
static std::vector<Box3D> decode_nms(hbDNNTensor* outputs, int n_out) {
    std::vector<Box3D> all_boxes;

    for (int h = 0; h < n_out && h < 6; h++) {
        const auto& props = outputs[h].properties;
        float* data = (float*)outputs[h].sysMem.virAddr;
        int n_cls = CLASSES_PER_HEAD[h];
        int H = props.validShape.dimensionSize[2];
        int W = props.validShape.dimensionSize[3];
        int c_stride = props.stride[1] / sizeof(float);
        int r_stride = props.stride[2] / sizeof(float);

        int off_hm=0, off_ctr=n_cls, off_cz=n_cls+2, off_dim=n_cls+3, off_rot=n_cls+6, off_vel=n_cls+8;

        auto at = [&](int ch, int gy, int gx) -> float {
            return data[ch*c_stride + gy*r_stride + gx];
        };

        for (int c = 0; c < n_cls; c++) {
            std::vector<std::pair<float,int>> scores;
            for (int gy = 0; gy < H; gy++) {
                const float* row = data + (off_hm+c)*c_stride + gy*r_stride;
                for (int gx = 0; gx < W; gx++) {
                    float s = 1.0f/(1.0f+expf(-row[gx]));
                    if (s > SCORE_THRESH) scores.push_back({s, gy*W+gx});
                }
            }
            std::sort(scores.begin(), scores.end(), [](auto& a, auto& b){ return a.first > b.first; });
            int K = std::min(TOP_K, (int)scores.size());

            for (int k = 0; k < K; k++) {
                int gy = scores[k].second/W, gx = scores[k].second%W;
                Box3D box;
                box.score = scores[k].first;
                box.cls = HEAD_CLS_OFFSET[h] + c;
                box.x = (gx + at(off_ctr+0,gy,gx)) * FEAT_STRIDE * VOXEL_SIZE[0] + PC_RANGE[0];
                box.y = (gy + at(off_ctr+1,gy,gx)) * FEAT_STRIDE * VOXEL_SIZE[1] + PC_RANGE[1];
                box.z = at(off_cz,gy,gx);
                box.w = expf(at(off_dim+0,gy,gx));
                box.l = expf(at(off_dim+1,gy,gx));
                box.h = expf(at(off_dim+2,gy,gx));
                box.yaw = atan2f(at(off_rot+1,gy,gx), at(off_rot+0,gy,gx));
                box.vx = at(off_vel+0,gy,gx);
                box.vy = at(off_vel+1,gy,gx);
                all_boxes.push_back(box);
            }
        }
    }

    std::unordered_map<int, std::vector<Box3D>> cls_boxes;
    for (auto& b : all_boxes) cls_boxes[b.cls].push_back(b);

    std::vector<std::thread> threads;
    std::mutex mu;
    std::vector<Box3D> final_boxes;
    for (auto& [cls, boxes] : cls_boxes) {
        threads.emplace_back([&boxes, &final_boxes, &mu]() {
            nms(boxes);
            std::lock_guard<std::mutex> lk(mu);
            for (auto& b : boxes) final_boxes.push_back(b);
        });
    }
    for (auto& t : threads) t.join();

    // Class NMS runs in parallel, so thread completion order is nondeterministic.
    // Use a total ordering to keep the emitted JSON byte-reproducible on ties.
    std::sort(final_boxes.begin(), final_boxes.end(), [](const Box3D& a, const Box3D& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.cls != b.cls) return a.cls < b.cls;
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        if (a.z != b.z) return a.z < b.z;
        if (a.w != b.w) return a.w < b.w;
        if (a.l != b.l) return a.l < b.l;
        if (a.h != b.h) return a.h < b.h;
        if (a.yaw != b.yaw) return a.yaw < b.yaw;
        if (a.vx != b.vx) return a.vx < b.vx;
        return a.vy < b.vy;
    });
    if ((int)final_boxes.size() > MAX_DETS_PER_SAMPLE)
        final_boxes.resize(MAX_DETS_PER_SAMPLE);
    return final_boxes;
}

// ============================================================
// Frame index (l2g uses double for precision)
// ============================================================
struct SweepInfo {
    std::string path;
    float rot[9], trans[3], time_lag;
};

struct FrameInfo {
    std::string token;
    std::string lidar_path;
    double l2g_rot[9], l2g_trans[3];
    std::vector<SweepInfo> sweeps;
};

static std::vector<FrameInfo> load_index(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open index %s\n", path); exit(1); }
    int n; fscanf(f, "%d", &n);
    std::vector<FrameInfo> frames(n);
    for (int i = 0; i < n; i++) {
        char tok[64], lp[512];
        fscanf(f, "%s %s", tok, lp);
        frames[i].token = tok;
        frames[i].lidar_path = lp;
        for (int j = 0; j < 9; j++) fscanf(f, "%lf", &frames[i].l2g_rot[j]);
        for (int j = 0; j < 3; j++) fscanf(f, "%lf", &frames[i].l2g_trans[j]);
        int ns; fscanf(f, "%d", &ns);
        frames[i].sweeps.resize(ns);
        for (int s = 0; s < ns; s++) {
            char sp[512];
            fscanf(f, "%s", sp);
            frames[i].sweeps[s].path = sp;
            for (int j = 0; j < 9; j++) fscanf(f, "%f", &frames[i].sweeps[s].rot[j]);
            for (int j = 0; j < 3; j++) fscanf(f, "%f", &frames[i].sweeps[s].trans[j]);
            fscanf(f, "%f", &frames[i].sweeps[s].time_lag);
        }
    }
    fclose(f);
    return frames;
}

// ============================================================
// Load keyframe + sweeps, apply transforms, merge
// ============================================================
static std::vector<float> load_frame(const FrameInfo& frame, const char* data_root, int& n_pts) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", data_root, frame.lidar_path.c_str());
    int kf_n;
    auto kf = load_points(path, kf_n);
    if (kf_n == 0) {
        fprintf(stderr, "WARN: missing keyframe %s\n", path);
        n_pts = 0;
        return {};
    }
    for (int i = 0; i < kf_n; i++) kf[i*5+4] = 0.0f;

    std::vector<float> merged = std::move(kf);
    // Reserve once so appending nine sweeps does not repeatedly reallocate and
    // copy all points already merged into the frame.
    merged.reserve((size_t)(kf_n + frame.sweeps.size() * 40000) * 5);
    n_pts = kf_n;

    for (auto& sw : frame.sweeps) {
        snprintf(path, sizeof(path), "%s/%s", data_root, sw.path.c_str());
        int sw_n;
        auto pts = load_points(path, sw_n);
        if (sw_n == 0) continue;
        const float* R = sw.rot;
        const float* T = sw.trans;
        for (int i = 0; i < sw_n; i++) {
            float x = pts[i*5], y = pts[i*5+1], z = pts[i*5+2];
            pts[i*5]   = R[0]*x + R[1]*y + R[2]*z + T[0];
            pts[i*5+1] = R[3]*x + R[4]*y + R[5]*z + T[1];
            pts[i*5+2] = R[6]*x + R[7]*y + R[8]*z + T[2];
            pts[i*5+4] = sw.time_lag;
        }
        merged.insert(merged.end(), pts.begin(), pts.end());
        n_pts += sw_n;
    }
    return merged;
}

// ============================================================
// Bounded asynchronous loader. It preserves frame order and exact float data,
// but overlaps SSD reads + sweep transforms for future frames with the current
// frame's VFE/BPU/NMS work.
// ============================================================
struct LoadedFrame {
    int index = -1;
    int n_pts = 0;
    double load_ms = 0.0;
    std::vector<float> points;
};

class FramePrefetcher {
public:
    FramePrefetcher(const std::vector<FrameInfo>& frames, int n_frames,
                    const char* data_root, size_t capacity)
        : frames_(frames), n_frames_(n_frames), data_root_(data_root),
          capacity_(capacity), worker_(&FramePrefetcher::run, this) {}

    ~FramePrefetcher() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        can_load_.notify_all();
        can_consume_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    bool pop(int expected_index, LoadedFrame& out) {
        std::unique_lock<std::mutex> lk(mu_);
        can_consume_.wait(lk, [&] { return stop_ || done_ || !queue_.empty(); });
        if (queue_.empty()) return false;
        out = std::move(queue_.front());
        queue_.pop_front();
        lk.unlock();
        can_load_.notify_one();
        if (out.index != expected_index) {
            fprintf(stderr, "Prefetch order error: expected %d, got %d\n",
                    expected_index, out.index);
            return false;
        }
        return true;
    }

    void join() {
        if (worker_.joinable()) worker_.join();
    }

private:
    void run() {
        for (int fi = 0; fi < n_frames_; ++fi) {
            LoadedFrame item;
            item.index = fi;
            Timer load_t; load_t.start();
            item.points = load_frame(frames_[fi], data_root_.c_str(), item.n_pts);
            item.load_ms = load_t.ms();

            std::unique_lock<std::mutex> lk(mu_);
            can_load_.wait(lk, [&] { return stop_ || queue_.size() < capacity_; });
            if (stop_) break;
            queue_.push_back(std::move(item));
            lk.unlock();
            can_consume_.notify_one();
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            done_ = true;
        }
        can_consume_.notify_all();
    }

    const std::vector<FrameInfo>& frames_;
    int n_frames_;
    std::string data_root_;
    size_t capacity_;
    std::deque<LoadedFrame> queue_;
    std::mutex mu_;
    std::condition_variable can_load_;
    std::condition_variable can_consume_;
    bool stop_ = false;
    bool done_ = false;
    std::thread worker_;
};

// ============================================================
// Quaternion helpers (double precision for global transform)
// ============================================================
static void rotmat_to_quat(const double R[9], double q[4]) {
    double tr = R[0]+R[4]+R[8];
    if (tr > 0) {
        double s = 2.0 * sqrt(tr+1.0);
        q[0]=0.25*s; q[1]=(R[7]-R[5])/s; q[2]=(R[2]-R[6])/s; q[3]=(R[3]-R[1])/s;
    } else if (R[0]>R[4] && R[0]>R[8]) {
        double s = 2.0 * sqrt(1.0+R[0]-R[4]-R[8]);
        q[0]=(R[7]-R[5])/s; q[1]=0.25*s; q[2]=(R[1]+R[3])/s; q[3]=(R[2]+R[6])/s;
    } else if (R[4]>R[8]) {
        double s = 2.0 * sqrt(1.0+R[4]-R[0]-R[8]);
        q[0]=(R[2]-R[6])/s; q[1]=(R[1]+R[3])/s; q[2]=0.25*s; q[3]=(R[5]+R[7])/s;
    } else {
        double s = 2.0 * sqrt(1.0+R[8]-R[0]-R[4]);
        q[0]=(R[3]-R[1])/s; q[1]=(R[2]+R[6])/s; q[2]=(R[5]+R[7])/s; q[3]=0.25*s;
    }
}

static void quat_mul(const double a[4], const double b[4], double c[4]) {
    c[0] = a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3];
    c[1] = a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2];
    c[2] = a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1];
    c[3] = a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0];
}

// ============================================================
// Transform box from lidar frame to global frame (double precision)
// ============================================================
struct Detection {
    std::string token;
    double tx, ty, tz;
    float sw, sl, sh;
    double qw, qx, qy, qz;
    double vx, vy;
    float score;
    int cls;
};

static Detection to_global(const Box3D& box, const double R[9], const double T[3], const char* token) {
    Detection d;
    d.token = token;
    d.tx = R[0]*box.x + R[1]*box.y + R[2]*box.z + T[0];
    d.ty = R[3]*box.x + R[4]*box.y + R[5]*box.z + T[1];
    d.tz = R[6]*box.x + R[7]*box.y + R[8]*box.z + T[2];
    d.sw = box.l; d.sl = box.w; d.sh = box.h;
    double q_R[4]; rotmat_to_quat(R, q_R);
    double q_box[4] = {cos(box.yaw/2.0), 0, 0, sin(box.yaw/2.0)};
    quat_mul(q_R, q_box, &d.qw);
    d.vx = R[0]*box.vx + R[1]*box.vy;
    d.vy = R[3]*box.vx + R[4]*box.vy;
    d.score = box.score;
    d.cls = box.cls;
    return d;
}

// ============================================================
// nuScenes attribute assignment (matches OpenPCDet)
// ============================================================
static const char* get_attr(int cls, double vx, double vy) {
    double vel = sqrt(vx*vx + vy*vy);
    if (cls <= 4) return vel > 0.2 ? "vehicle.moving" : "vehicle.parked";
    if (cls == 8) return vel > 0.2 ? "pedestrian.moving" : "pedestrian.standing";
    if (cls == 6 || cls == 7) return vel > 0.2 ? "cycle.with_rider" : "cycle.without_rider";
    return "";
}

// ============================================================
// Write one detection to JSON
// ============================================================
static void write_det(FILE* f, const Detection& d, bool comma) {
    fprintf(f, "%s{\"sample_token\":\"%s\","
        "\"translation\":[%.4f,%.4f,%.4f],"
        "\"size\":[%.4f,%.4f,%.4f],"
        "\"rotation\":[%.6f,%.6f,%.6f,%.6f],"
        "\"velocity\":[%.4f,%.4f],"
        "\"detection_name\":\"%s\","
        "\"detection_score\":%.6f,"
        "\"attribute_name\":\"%s\"}",
        comma ? "," : "",
        d.token.c_str(),
        d.tx, d.ty, d.tz,
        d.sw, d.sl, d.sh,
        d.qw, d.qx, d.qy, d.qz,
        d.vx, d.vy,
        DET_NAMES[d.cls],
        d.score,
        get_attr(d.cls, d.vx, d.vy));
}

// ============================================================
// Process one frame's BPU output → transform → write JSON
// ============================================================
static void process_output(hbDNNTensor* out, int n_out,
                           const FrameInfo& frame, FILE* jf,
                           bool comma, int& total_dets) {
    auto boxes = decode_nms(out, n_out);
    if (comma) fprintf(jf, ",");
    fprintf(jf, "\"%s\":[", frame.token.c_str());
    for (size_t i = 0; i < boxes.size(); i++) {
        auto det = to_global(boxes[i], frame.l2g_rot, frame.l2g_trans, frame.token.c_str());
        write_det(jf, det, i > 0);
    }
    fprintf(jf, "]");
    total_dets += (int)boxes.size();
}

// ============================================================
// Main — pipelined:
//   loader(N+2) || CPU(NMS(N-1) + VFE(N+1)) || BPU(N)
// ============================================================
int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <model.hbm> <vfe_dir> <index.txt> <data_root> <output.json> [max_frames] [serial]\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* vfe_dir = argv[2];
    const char* index_path = argv[3];
    const char* data_root = argv[4];
    const char* output_path = argv[5];
    int max_frames = argc >= 7 ? atoi(argv[6]) : 0;
    bool serial = argc >= 8 && atoi(argv[7]) != 0;

    setbuf(stdout, NULL);

    load_vfe_weights(vfe_dir);
    printf("VFE weights loaded from %s\n", vfe_dir);

    auto frames = load_index(index_path);
    int n_frames = (int)frames.size();
    if (max_frames > 0 && max_frames < n_frames) n_frames = max_frames;
    printf("Index: %d frames (processing %d)\n", (int)frames.size(), n_frames);

    // Load BPU model
    hbDNNPackedHandle_t packed;
    hbDNNHandle_t handle;
    const char* files[] = {model_path};
    CHECK(hbDNNInitializeFromFiles(&packed, files, 1));
    const char** names; int32_t cnt;
    CHECK(hbDNNGetModelNameList(&names, &cnt, packed));
    CHECK(hbDNNGetModelHandle(&handle, packed, names[0]));
    int n_in, n_out;
    CHECK(hbDNNGetInputCount(&n_in, handle));
    CHECK(hbDNNGetOutputCount(&n_out, handle));
    printf("Model: %s, inputs=%d, outputs=%d\n", model_path, n_in, n_out);

    // Allocate double-buffered BPU tensors
    hbDNNTensor inputs_A[2], outputs_A[32];
    hbDNNTensor inputs_B[2], outputs_B[32];
    for (int i = 0; i < n_in; i++) {
        CHECK(hbDNNGetInputTensorProperties(&inputs_A[i].properties, handle, i));
        CHECK(hbUCPMallocCached(&inputs_A[i].sysMem, inputs_A[i].properties.alignedByteSize, 0));
        inputs_B[i].properties = inputs_A[i].properties;
        CHECK(hbUCPMallocCached(&inputs_B[i].sysMem, inputs_A[i].properties.alignedByteSize, 0));
    }
    for (int i = 0; i < n_out; i++) {
        CHECK(hbDNNGetOutputTensorProperties(&outputs_A[i].properties, handle, i));
        CHECK(hbUCPMallocCached(&outputs_A[i].sysMem, outputs_A[i].properties.alignedByteSize, 0));
        outputs_B[i].properties = outputs_A[i].properties;
        CHECK(hbUCPMallocCached(&outputs_B[i].sysMem, outputs_A[i].properties.alignedByteSize, 0));
    }

    // Warmup
    memset(inputs_A[0].sysMem.virAddr, 0, inputs_A[0].properties.alignedByteSize);
    if (n_in > 1) memset(inputs_A[1].sysMem.virAddr, 0, inputs_A[1].properties.alignedByteSize);
    {
        hbUCPTaskHandle_t wtask = nullptr;
        for (int i = 0; i < n_in; i++) CHECK(hbUCPMemFlush(&inputs_A[i].sysMem, HB_SYS_MEM_CACHE_CLEAN));
        CHECK(hbDNNInferV2(&wtask, outputs_A, inputs_A, handle));
        hbUCPSchedParam sp; memset(&sp, 0, sizeof(sp));
        CHECK(hbUCPSubmitTask(wtask, &sp));
        CHECK(hbUCPWaitTaskDone(wtask, 0));
        CHECK(hbUCPReleaseTask(wtask));
    }
    printf("Warmup done.\n\n");

    FILE* jf = fopen(output_path, "w");
    if (!jf) { fprintf(stderr, "Cannot open %s for writing\n", output_path); return 1; }
    fprintf(jf, "{\"meta\":{\"use_camera\":false,\"use_lidar\":true,\"use_radar\":false,\"use_map\":false,\"use_external\":false},\"results\":{");

    Timer total_timer; total_timer.start();
    int total_dets = 0;

    // Per-frame timing log
    char timing_path[1024];
    snprintf(timing_path, sizeof(timing_path), "%s.timing.csv", output_path);
    FILE* tf = fopen(timing_path, "w");
    if (tf) fprintf(tf, "%s\n", serial ? "frame,load_ms,vfe_ms,bpu_ms,decode_nms_ms,total_ms"
                                       : "frame,decode_nms_ms,load_vfe_ms,prefetch_load_ms,prefetch_wait_ms,vfe_ms,bpu_submit_done_ms,wall_ms");

    // ---------------- Serial mode: true single-frame latency ----------------
    if (serial) {
        for (int fi = 0; fi < n_frames; fi++) {
            Timer t_all; t_all.start();

            Timer t; t.start();
            int n_pts;
            auto pts = load_frame(frames[fi], data_root, n_pts);
            double load_ms = t.ms();

            t.start();
            if (n_pts > 0)
                real_vfe(pts.data(), n_pts,
                         (float*)inputs_A[0].sysMem.virAddr,
                         n_in > 1 ? (float*)inputs_A[1].sysMem.virAddr : nullptr);
            double vfe_ms = t.ms();

            for (int i = 0; i < n_in; i++)
                CHECK(hbUCPMemFlush(&inputs_A[i].sysMem, HB_SYS_MEM_CACHE_CLEAN));
            hbUCPTaskHandle_t task = nullptr;
            CHECK(hbDNNInferV2(&task, outputs_A, inputs_A, handle));
            hbUCPSchedParam sp; memset(&sp, 0, sizeof(sp));
            t.start();
            CHECK(hbUCPSubmitTask(task, &sp));
            CHECK(hbUCPWaitTaskDone(task, 0));
            CHECK(hbUCPReleaseTask(task));
            double bpu_ms = t.ms();

            for (int i = 0; i < n_out; i++)
                CHECK(hbUCPMemFlush(&outputs_A[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE));
            t.start();
            process_output(outputs_A, n_out, frames[fi], jf, fi > 0, total_dets);
            double nms_ms = t.ms();

            double total_ms = t_all.ms();
            if (tf) fprintf(tf, "%d,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                            fi, load_ms, vfe_ms, bpu_ms, nms_ms, total_ms);
            if (fi % 10 == 0 || fi == n_frames - 1)
                printf("[serial %d/%d] %.1f ms (bpu %.1f)\n", fi + 1, n_frames, total_ms, bpu_ms);
        }
    }

    // ---------------- Pipelined mode (asynchronous loader, instrumented) ----------------
    if (!serial) {
    struct PipeTiming {
        double decode_nms_ms = 0.0;
        double prefetch_load_ms = 0.0;
        double prefetch_wait_ms = 0.0;
        double vfe_ms = 0.0;
        double bpu_submit_done_ms = 0.0;
        double wall_ms = 0.0;
    };
    std::vector<PipeTiming> pipe_timing(n_frames);
    FramePrefetcher prefetcher(frames, n_frames, data_root, 2);
    printf("Async frame prefetch enabled (depth=2).\n");

    // Prepare frame 0 into buffer A
    {
        Timer wait_t; wait_t.start();
        LoadedFrame loaded;
        if (!prefetcher.pop(0, loaded)) {
            fprintf(stderr, "Failed to prefetch frame 0\n");
            return 1;
        }
        pipe_timing[0].prefetch_wait_ms = wait_t.ms();
        pipe_timing[0].prefetch_load_ms = loaded.load_ms;
        Timer vfe_t; vfe_t.start();
        if (loaded.n_pts > 0)
            real_vfe(loaded.points.data(), loaded.n_pts,
                     (float*)inputs_A[0].sysMem.virAddr,
                     n_in > 1 ? (float*)inputs_A[1].sysMem.virAddr : nullptr);
        pipe_timing[0].vfe_ms = vfe_t.ms();
    }

    hbDNNTensor* prev_out = nullptr;
    int prev_fi = -1;

    for (int fi = 0; fi < n_frames; fi++) {
        Timer iter_t; iter_t.start();
        hbDNNTensor* cur_in  = (fi%2==0) ? inputs_A  : inputs_B;
        hbDNNTensor* cur_out = (fi%2==0) ? outputs_A : outputs_B;
        hbDNNTensor* nxt_in  = (fi%2==0) ? inputs_B  : inputs_A;

        // Submit BPU for current frame
        for (int i = 0; i < n_in; i++)
            CHECK(hbUCPMemFlush(&cur_in[i].sysMem, HB_SYS_MEM_CACHE_CLEAN));
        hbUCPTaskHandle_t task = nullptr;
        CHECK(hbDNNInferV2(&task, cur_out, cur_in, handle));
        hbUCPSchedParam sp; memset(&sp, 0, sizeof(sp));
        Timer sub_t; sub_t.start();
        CHECK(hbUCPSubmitTask(task, &sp));

        // While BPU runs: process previous + prepare next
        Timer cpu_t; cpu_t.start();
        if (prev_out && prev_fi >= 0) {
            for (int i = 0; i < n_out; i++)
                CHECK(hbUCPMemFlush(&prev_out[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE));
            process_output(prev_out, n_out, frames[prev_fi], jf, prev_fi > 0, total_dets);
        }
        double decode_ms = cpu_t.ms();
        if (prev_fi >= 0) pipe_timing[prev_fi].decode_nms_ms = decode_ms;

        Timer wait_t; wait_t.start();
        double prefetch_wait_ms = 0.0;
        double vfe_ms = 0.0;
        if (fi + 1 < n_frames) {
            LoadedFrame loaded;
            if (!prefetcher.pop(fi + 1, loaded)) {
                fprintf(stderr, "Failed to prefetch frame %d\n", fi + 1);
                return 1;
            }
            prefetch_wait_ms = wait_t.ms();
            pipe_timing[fi + 1].prefetch_load_ms = loaded.load_ms;
            pipe_timing[fi + 1].prefetch_wait_ms = prefetch_wait_ms;
            Timer vfe_t; vfe_t.start();
            if (loaded.n_pts > 0)
                real_vfe(loaded.points.data(), loaded.n_pts,
                         (float*)nxt_in[0].sysMem.virAddr,
                         n_in > 1 ? (float*)nxt_in[1].sysMem.virAddr : nullptr);
            vfe_ms = vfe_t.ms();
            pipe_timing[fi + 1].vfe_ms = vfe_ms;
        }

        // Wait BPU
        CHECK(hbUCPWaitTaskDone(task, 0));
        double bpu_wait_ms = sub_t.ms();
        CHECK(hbUCPReleaseTask(task));
        pipe_timing[fi].bpu_submit_done_ms = bpu_wait_ms;
        pipe_timing[fi].wall_ms = iter_t.ms();

        prev_out = cur_out;
        prev_fi = fi;

        if (fi % 100 == 0 || fi == n_frames-1) {
            double elapsed = total_timer.ms() / 1000.0;
            double eta = (fi > 0) ? elapsed / fi * (n_frames - fi) : 0;
            printf("[%d/%d] elapsed %.0fs, ETA %.0fs, %.1f ms/frame\n",
                   fi+1, n_frames, elapsed, eta, fi > 0 ? elapsed*1000.0/fi : 0);
        }
    }

    // Process last frame
    if (prev_out && prev_fi >= 0) {
        Timer decode_t; decode_t.start();
        for (int i = 0; i < n_out; i++)
            CHECK(hbUCPMemFlush(&prev_out[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE));
        process_output(prev_out, n_out, frames[prev_fi], jf, prev_fi > 0, total_dets);
        pipe_timing[prev_fi].decode_nms_ms = decode_t.ms();
    }
    prefetcher.join();
    if (tf) {
        for (int fi = 0; fi < n_frames; ++fi) {
            const auto& t = pipe_timing[fi];
            const double load_vfe_ms = t.prefetch_load_ms + t.vfe_ms;
            fprintf(tf, "%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                    fi, t.decode_nms_ms, load_vfe_ms, t.prefetch_load_ms,
                    t.prefetch_wait_ms, t.vfe_ms, t.bpu_submit_done_ms, t.wall_ms);
        }
    }
    } // end if (!serial)

    fprintf(jf, "}}");
    fclose(jf);

    double total_s = total_timer.ms() / 1000.0;
    printf("\nDone: %d frames, %d total detections, %.1f s (%.1f ms/frame)\n",
           n_frames, total_dets, total_s, total_s*1000.0/n_frames);
    printf("Output: %s\n", output_path);
    printf("Timing log: %s\n", timing_path);
    printf("Peak RSS (VmHWM): %ld kB\n", read_vmhwm_kb());
    if (tf) fclose(tf);

    for (int i = 0; i < n_in; i++) { hbUCPFree(&inputs_A[i].sysMem); hbUCPFree(&inputs_B[i].sysMem); }
    for (int i = 0; i < n_out; i++) { hbUCPFree(&outputs_A[i].sysMem); hbUCPFree(&outputs_B[i].sysMem); }
    hbDNNRelease(packed);
    return 0;
}
