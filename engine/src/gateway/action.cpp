// gateway/action.cpp — ActionType enum ↔ string helpers.
// see docs/knowledge-base/part4-gateway-execution.md §2.24–§2.26

#include "juniauto/gateway/decision.hpp"

namespace juniauto::gateway {

// String representation of ActionType (matches schema.sql action_type values).
// Used by QuestDB write path and logging; not in the public header (internal linkage).
const char* action_type_str(ActionType a) noexcept {
    switch (a) {
        case ActionType::Buy:     return "BUY";
        case ActionType::Sell:    return "SELL";
        case ActionType::Rotate:  return "ROTATE";
        case ActionType::Replace: return "REPLACE";
        case ActionType::Cancel:  return "CANCEL";
        case ActionType::Hold:    return "HOLD";
    }
    return "UNKNOWN";
}

}  // namespace juniauto::gateway
