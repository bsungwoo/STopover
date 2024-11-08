#include "utils.h"

namespace STopoverUtils {

/**
 * @brief Checks if all elements of `subset` are present in `superset`.
 *
 * @param subset The subset vector.
 * @param superset The superset vector.
 * @return true if `subset` is entirely contained within `superset`, false otherwise.
 */
bool is_subset(const std::vector<int>& subset, const std::vector<int>& superset) {
    return std::all_of(subset.begin(), subset.end(), [&](int x) {
        return std::find(superset.begin(), superset.end(), x) != superset.end();
    });
}

/**
 * @brief Computes the size of the intersection between two vectors.
 *
 * @param a The first vector.
 * @param b The second vector.
 * @return The number of elements common to both vectors.
 */
size_t intersection_size(const std::vector<int>& a, const std::vector<int>& b) {
    std::set<int> set_a(a.begin(), a.end());
    std::set<int> set_b(b.begin(), b.end());
    std::vector<int> intersection;
    std::set_intersection(set_a.begin(), set_a.end(),
                          set_b.begin(), set_b.end(),
                          std::back_inserter(intersection));
    return intersection.size();
}

} // namespace STopoverUtils