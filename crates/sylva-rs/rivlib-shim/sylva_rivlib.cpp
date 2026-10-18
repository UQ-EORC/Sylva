// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//
// sylva_rivlib: a small C interface over RiVLib's C++ `pointcloud` class.
//
// RiVLib's C API (libscanifc) delivers echoes only. Reproducing RiSCAN PRO's
// scan conversion needs what the C++ class knows in addition: every laser
// shot (also the ~12% without echoes) with its exact time and beam geometry,
// scan line boundaries, and the echo ranges under each multiple-time-around
// zone hypothesis. This shim streams shots and echoes in shot order through
// a C API that Sylva loads at run time with libloading.
//
// Build (RiVLib is proprietary; it is linked statically into this library and
// the result stays on your machine):  see build.sh next to this file.
//
// Late echoes: an echo whose time of flight exceeds the pulse interval is
// delivered by RiVLib after on_shot() of the following pulse. It belongs to
// the pulse whose time equals target.time, so echoes are assigned to shots by
// that timestamp through a ring of recent shots.
#include <riegl/scanlib.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace scanlib;

extern "C" {

#define SYLVA_RIVLIB_API_VERSION 1
#define SYLVA_MAX_ZONE 8

#pragma pack(push, 1)
// One laser shot. `line` counts scan lines from the first line_start packet,
// `shot` counts shots within the line, both 0-based. Shots before the first
// line start are not reported.
typedef struct {
    double   time;            // pulse time [s], scanner internal clock
    uint32_t line;
    uint32_t shot;
    uint8_t  facet;
    uint8_t  flags;           // bit4: line is a segment line
    uint8_t  n_echo;          // echoes of this shot that follow in the echo buffer
    uint8_t  pad;
    float    ox, oy, oz;      // beam origin [m]
    float    dx, dy, dz;      // unit beam direction
} sylva_shot;

// One echo, listed after its shot in shot order.
typedef struct {
    double   range;           // echo_range in the standard zone (zone 1) [m]
    float    amplitude;       // [dB]
    float    reflectance;     // [dB], for the zone-1 range
    float    deviation;
    float    background;      // background radiation (NaN if not available)
    uint16_t flags;           // bit0 is_sw, bit1 high_power, bit2 rising_edge, bit3 pps_locked, bit5 late echo,
                              // bits 6-7 echo type (0 single, 1 first, 2 interior, 3 last)
    uint16_t nzone;           // hypotheses filled below
    float    zone_range[SYLVA_MAX_ZONE];       // echo_range if the echo were in zone k+1 (NaN: none)
    float    zone_reflectance[SYLVA_MAX_ZONE]; // reflectance for that range
} sylva_echo;
#pragma pack(pop)

struct shot_entry {
    sylva_shot shot;
    std::vector<sylva_echo> echoes;
};

struct reader : public pointcloud {
    std::shared_ptr<basic_rconnection> rc;
    std::unique_ptr<decoder_rxpmarker> dec;
    buffer buf;
    bool opened = false, eoi = false;
    std::string error;
    unsigned max_zone;
    uint32_t line = 0, shot_in_line = 0;
    bool line_started = false;
    uint8_t seg_flag = 0;
    std::deque<shot_entry> ring;            // shots still receiving echoes (newest last)
    std::deque<shot_entry> done;            // completed shots waiting to be read
    static const size_t RING = 8;
    uint64_t n_shots = 0, n_echoes = 0, n_late = 0, n_orphan = 0;
    bool have_notch = false;
    float notch_min = 0, notch_max = 0, notch_amp = 0;   // the RXP's notch packet (window echo gate)

    reader(unsigned max_zone_) : pointcloud(false), max_zone(max_zone_) {}

    void on_notch_filter(const notch_filter<iterator_type>& a) override {
        pointcloud::on_notch_filter(a);
        have_notch = true; notch_min = a.range_min; notch_max = a.range_max; notch_amp = a.amplitude_max;
    }

    void new_line(uint8_t seg) {
        if (line_started) ++line; else line_started = true;
        shot_in_line = 0; seg_flag = seg;
    }
    void on_line_start_up(const line_start_up<iterator_type>& a) override { pointcloud::on_line_start_up(a); new_line(0); }
    void on_line_start_dn(const line_start_dn<iterator_type>& a) override { pointcloud::on_line_start_dn(a); new_line(0); }
    void on_line_start_segment_1(const line_start_segment_1<iterator_type>& a) override { pointcloud::on_line_start_segment_1(a); new_line(16); }
    void on_line_start_segment_2(const line_start_segment_2<iterator_type>& a) override { pointcloud::on_line_start_segment_2(a); new_line(16); }
    void on_line_start_segment_3(const line_start_segment_3<iterator_type>& a) override { pointcloud::on_line_start_segment_3(a); new_line(16); }

    void on_shot() override {
        pointcloud::on_shot();
        if (line_started) {
            shot_entry s;
            memset(&s.shot, 0, sizeof s.shot);
            s.shot.time = time; s.shot.line = line; s.shot.shot = shot_in_line;
            s.shot.facet = (uint8_t)facet; s.shot.flags = seg_flag;
            s.shot.ox = (float)beam_origin[0]; s.shot.oy = (float)beam_origin[1]; s.shot.oz = (float)beam_origin[2];
            s.shot.dx = (float)beam_direction[0]; s.shot.dy = (float)beam_direction[1]; s.shot.dz = (float)beam_direction[2];
            ring.push_back(std::move(s));
            ++n_shots;
            while (ring.size() > RING) { done.push_back(std::move(ring.front())); ring.pop_front(); }
        }
        ++shot_in_line;
    }

    void on_echo_transformed(echo_type e) override {
        pointcloud::on_echo_transformed(e);
        const target& t = targets[target_count - 1];
        shot_entry* s = nullptr; bool late = false;
        for (size_t k = ring.size(); k-- > 0;) {
            if (ring[k].shot.time == t.time) { s = &ring[k]; late = (k + 1 != ring.size()); break; }
        }
        if (!s) { ++n_orphan; return; }
        if (late) ++n_late;
        sylva_echo r; memset(&r, 0, sizeof r);
        r.range = t.echo_range;
        r.amplitude = t.amplitude; r.reflectance = t.reflectance; r.deviation = t.deviation; r.background = t.background_radiation;
        r.flags = (t.is_sw ? 1 : 0) | (t.is_high_power ? 2 : 0) | (t.is_rising_edge ? 4 : 0) | (t.is_pps_locked ? 8 : 0) | (late ? 32 : 0)
                | (uint16_t)(((unsigned)e & 3u) << 6);
        size_t nz = multi_mta_targets.size() < (size_t)SYLVA_MAX_ZONE ? multi_mta_targets.size() : (size_t)SYLVA_MAX_ZONE;
        r.nzone = (uint16_t)nz;
        for (size_t k = 0; k < (size_t)SYLVA_MAX_ZONE; ++k) {
            r.zone_range[k] = std::numeric_limits<float>::quiet_NaN();
            r.zone_reflectance[k] = std::numeric_limits<float>::quiet_NaN();
        }
        // the zone lists hold the same echoes in the same order as `targets`
        size_t i = target_count - 1;
        for (size_t k = 0; k < nz; ++k) {
            const std::vector<target>& zl = multi_mta_targets[k];
            if (i < zl.size() && std::fabs(zl[i].amplitude - t.amplitude) < 1e-3f) {
                r.zone_range[k] = (float)zl[i].echo_range;
                r.zone_reflectance[k] = zl[i].reflectance;
            }
        }
        if (s->echoes.size() < 255) { s->echoes.push_back(r); ++s->shot.n_echo; ++n_echoes; }
    }

    // Decode until at least `want` complete shots are available or the file ends.
    bool pump(size_t want) {
        try {
            while (!eoi && done.size() < want) {
                dec->get(buf);
                if (dec->eoi()) {
                    eoi = true;
                    while (!ring.empty()) { done.push_back(std::move(ring.front())); ring.pop_front(); }
                    break;
                }
                dispatch(buf.begin(), buf.end());
            }
            return true;
        } catch (std::exception& e) {
            error = e.what();
            return false;
        }
    }
};

// Open an RXP (uri such as "file:/path/scan.rxp"). max_zone: zone hypotheses
// 1..max_zone are requested (1..8). apply_notch: 0 disables RiVLib's notch
// filter (recommended: RiSCAN applies the notch after MTA resolution).
int sylva_rxp_open(const char* uri, int max_zone, int apply_notch, void** handle) {
    if (!uri || !handle) return -1;
    if (max_zone < 1) max_zone = 1;
    if (max_zone > SYLVA_MAX_ZONE) max_zone = SYLVA_MAX_ZONE;
    reader* r = new reader((unsigned)max_zone);
    try {
        r->rc = basic_rconnection::create(uri);
        r->rc->open();
        r->dec.reset(new decoder_rxpmarker(r->rc));
        std::vector<unsigned> zones;
        for (unsigned z = 1; z <= (unsigned)max_zone; ++z) zones.push_back(z);
        r->set_multi_mta_zones(zones, 1);       // standard zone must be given explicitly
        if (!apply_notch) r->set_notch_filter(0.f, 0.f, 0.f);
        r->opened = true;
    } catch (std::exception& e) {
        r->error = e.what();
        *handle = r;
        return 1;
    }
    *handle = r;
    return 0;
}

// Read up to `max_shots` complete shots and their echoes. Echoes of shot i
// follow those of shot i-1; `shots[i].n_echo` says how many. A shot whose
// echoes would not fit `echo_cap` is held back for the next call (so make
// echo_cap >= 255). *eof is set when nothing more will come.
int sylva_rxp_read(void* handle, uint32_t max_shots, sylva_shot* shots, uint32_t* n_shots,
                   sylva_echo* echoes, uint32_t echo_cap, uint32_t* n_echoes, int* eof) {
    reader* r = (reader*)handle;
    if (!r || !r->opened) return -1;
    *n_shots = 0; *n_echoes = 0; *eof = 0;
    if (!r->pump(max_shots)) return 2;
    uint32_t ns = 0, ne = 0;
    while (ns < max_shots && !r->done.empty()) {
        shot_entry& s = r->done.front();
        if (ne + s.echoes.size() > echo_cap) break;
        shots[ns] = s.shot;
        if (!s.echoes.empty()) memcpy(echoes + ne, s.echoes.data(), s.echoes.size() * sizeof(sylva_echo));
        ne += (uint32_t)s.echoes.size();
        ++ns;
        r->done.pop_front();
    }
    *n_shots = ns; *n_echoes = ne;
    if (r->eoi && r->done.empty()) *eof = 1;
    return 0;
}

// Instrument / stream metadata as JSON (RiVLib's write_meta plus the fields
// Sylva needs at the top level). Call after the first read so the header
// packets have been seen. Returns the number of bytes needed; copies min(cap).
uint32_t sylva_rxp_meta(void* handle, char* buf, uint32_t cap) {
    reader* r = (reader*)handle;
    if (!r) return 0;
    std::ostringstream m;
    m << "{\"api_version\":" << SYLVA_RIVLIB_API_VERSION
      << ",\"type_id\":\"" << r->type_id << "\",\"serial\":\"" << r->serial << "\""
      << ",\"group_velocity\":" << r->group_velocity << ",\"unambiguous_range\":" << r->unambiguous_range
      << ",\"num_facets\":" << r->num_facets << ",\"max_zone\":" << r->max_zone
      << ",\"notch\":" << (r->have_notch ? "{" : "null")
      ;
    if (r->have_notch) m << "\"range_min\":" << r->notch_min << ",\"range_max\":" << r->notch_max << ",\"amplitude_max\":" << r->notch_amp << "}";
    m << ",\"n_shots\":" << r->n_shots << ",\"n_echoes\":" << r->n_echoes << ",\"n_late\":" << r->n_late << ",\"n_orphan\":" << r->n_orphan
      << ",\"rivlib\":";
    try { r->write_meta(m, false); } catch (...) { m << "null"; }
    m << "}";
    std::string s = m.str();
    if (buf && cap) { size_t n = s.size() < cap - 1 ? s.size() : cap - 1; memcpy(buf, s.data(), n); buf[n] = 0; }
    return (uint32_t)s.size() + 1;
}

// Message of the last error on this handle (empty string if none).
const char* sylva_rxp_last_error(void* handle) {
    reader* r = (reader*)handle;
    return r ? r->error.c_str() : "";
}

void sylva_rxp_close(void* handle) {
    reader* r = (reader*)handle;
    delete r;
}

int sylva_rxp_api_version(void) { return SYLVA_RIVLIB_API_VERSION; }

}  // extern "C"
