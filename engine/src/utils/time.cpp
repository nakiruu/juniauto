// time.cpp — trading-session time utilities.
// see docs/knowledge-base/part4-gateway-execution.md §2.24

#include "juniauto/utils/time.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace juniauto::time_util {

// Quote age in trading sessions = age_in_minutes / 390.
// IEX 15-min delay floor: 15 / 390 = 0.038 sessions, always < 0.5 threshold,
// so IEX quotes from the current session always return 0 stale-quote risk.
// see docs/knowledge-base/part4-gateway-execution.md §2.24 (stale-quote table)
double quote_age_sessions(Timestamp now, Timestamp quote_ts) noexcept {
    // IEX mandatory delay floor: 15 minutes.
    // see docs/knowledge-base/part4-gateway-execution.md §2.24
    constexpr double IEX_FLOOR_MINUTES = 15.0;

    if (now <= quote_ts) {
        // Clock skew or same-tick: apply IEX floor.
        return IEX_FLOOR_MINUTES / static_cast<double>(kMinutesPerTradingDay);
    }

    const auto delta = now - quote_ts;
    const double age_minutes =
        std::chrono::duration<double, std::ratio<60>>(delta).count();

    // Enforce the IEX minimum floor so that sub-15-min deltas are never reported as 0.
    const double floored_minutes = std::max(age_minutes, IEX_FLOOR_MINUTES);

    return floored_minutes / static_cast<double>(kMinutesPerTradingDay);
}

std::string session_name(Session s) {
    switch (s) {
        case Session::Regular:    return "regular";
        case Session::Premarket:  return "premarket";
        case Session::AfterHours: return "after_hours";
        case Session::Closed:     return "closed";
    }
    return "unknown";
}

}  // namespace juniauto::time_util
