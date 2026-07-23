// time.hpp — trading-day / session utilities on the C++ side.
// PRINCIPLESLONG.md references inline.

#pragma once

#include <chrono>
#include <string>

namespace juniauto::time_util {

using Clock = std::chrono::system_clock;
using Timestamp = std::chrono::time_point<Clock>;

constexpr int kMinutesPerTradingDay = 390;  // (§2.24)

enum class Session : int {
    Regular = 0,
    Premarket = 1,
    AfterHours = 2,
    Closed = 3,
};

// Session multiplier used in base_side_cost (§2.24).
inline double session_multiplier(Session s) noexcept {
    switch (s) {
        case Session::Regular:    return 1.0;
        case Session::Premarket:  return 1.5;
        case Session::AfterHours: return 2.0;
        case Session::Closed:     return 2.5;
    }
    return 1.0;
}

// Quote age in trading sessions given (now, quote_ts) both in UTC. Enforces IEX floor of 15 min.
double quote_age_sessions(Timestamp now, Timestamp quote_ts) noexcept;

// Adverse-selection share used in exit_cost_explicit_bps (§2.24).
inline double adverse_selection_share(Session s) noexcept {
    switch (s) {
        case Session::Regular:    return 0.35;
        case Session::Premarket:  return 0.55;
        case Session::AfterHours: return 0.55;
        case Session::Closed:     return 0.90;
    }
    return 0.35;
}

std::string session_name(Session s);

}  // namespace juniauto::time_util
