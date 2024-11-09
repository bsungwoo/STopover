#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <algorithm>
#include <set>

namespace STopoverUtils {

/**
 * @brief Checks if all elements of `subset` are present in `superset`.
 *
 * @param subset The subset vector.
 * @param superset The superset vector.
 * @return true if `subset` is entirely contained within `superset`, false otherwise.
 */
inline bool is_subset(const std::vector<int>& subset, const std::vector<int>& superset);
} // namespace STopoverUtils

#endif // UTILS_H