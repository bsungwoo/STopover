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
bool is_subset(const std::vector<int>& subset, const std::vector<int>& superset);

/**
 * @brief Computes the size of the intersection between two vectors.
 *
 * @param a The first vector.
 * @param b The second vector.
 * @return The number of elements common to both vectors.
 */
size_t intersection_size(const std::vector<int>& a, const std::vector<int>& b);

} // namespace STopoverUtils

#endif // UTILS_H