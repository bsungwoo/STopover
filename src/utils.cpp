#include "utils.h"
#include <vector>
#include <algorithm>

namespace STopoverUtils {

/**
 * @brief Checks if all elements of `subset` are present in `superset`.
 *
 * @param subset The subset vector.
 * @param superset The superset vector.
 * @return true if `subset` is entirely contained within `superset`, false otherwise.
 */
inline bool is_subset(const std::vector<int>& subset, const std::vector<int>& superset) {
    return std::all_of(subset.begin(), subset.end(), [&](int x) {
        return std::find(superset.begin(), superset.end(), x) != superset.end();
    });
}

} // namespace STopoverUtils