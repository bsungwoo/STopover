#include "jaccard.h"
#include "logging.h"
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <cmath>

// Function to calculate the Jaccard composite index from connected component locations
double jaccard_composite(const std::vector<std::vector<int>>& cc_1, 
                         const std::vector<std::vector<int>>& cc_2,
                         const std::string& jaccard_type) {
    try {
        log_message("jaccard_composite: Starting with " + 
                   std::to_string(cc_1.size()) + " cc_1 components, " +
                   std::to_string(cc_2.size()) + " cc_2 components, " +
                   "jaccard_type=" + jaccard_type);
        
        // Check for empty inputs
        if (cc_1.empty() || cc_2.empty()) {
            log_message("jaccard_composite: Empty input, returning 0");
            return 0.0;
        }
        
        // Check for invalid dimensions
        for (const auto& row : cc_1) {
            if (row.empty()) {
                log_message("jaccard_composite: Empty row in cc_1, returning 0");
                return 0.0;
            }
        }
        
        for (const auto& row : cc_2) {
            if (row.empty()) {
                log_message("jaccard_composite: Empty row in cc_2, returning 0");
                return 0.0;
            }
        }
        
        // Simplified Jaccard calculation
        if (jaccard_type == "default" || jaccard_type == "composite") {
            // Count unique elements in each set
            std::unordered_set<int> set1, set2, intersection;
            
            // Process cc_1
            for (const auto& row : cc_1) {
                for (int val : row) {
                    if (val > 0) {
                        set1.insert(val);
                    }
                }
            }
            
            // Process cc_2
            for (const auto& row : cc_2) {
                for (int val : row) {
                    if (val > 0) {
                        set2.insert(val);
                        if (set1.find(val) != set1.end()) {
                            intersection.insert(val);
                        }
                    }
                }
            }
            
            // Calculate Jaccard index
            double union_size = set1.size() + set2.size() - intersection.size();
            if (union_size > 0) {
                double jaccard = static_cast<double>(intersection.size()) / union_size;
                log_message("jaccard_composite: Calculated jaccard = " + std::to_string(jaccard));
                return jaccard;
            } else {
                log_message("jaccard_composite: Union size is 0, returning 0");
                return 0.0;
            }
        } 
        else if (jaccard_type == "weighted") {
            // Simplified weighted Jaccard
            log_message("jaccard_composite: Using weighted Jaccard");
            
            // Count elements with weights
            std::unordered_map<int, double> weights1, weights2;
            double total_weight1 = 0.0, total_weight2 = 0.0;
            
            // Process cc_1
            for (const auto& row : cc_1) {
                for (int val : row) {
                    if (val > 0) {
                        weights1[val] += 1.0;
                        total_weight1 += 1.0;
                    }
                }
            }
            
            // Process cc_2
            for (const auto& row : cc_2) {
                for (int val : row) {
                    if (val > 0) {
                        weights2[val] += 1.0;
                        total_weight2 += 1.0;
                    }
                }
            }
            
            // Normalize weights
            if (total_weight1 > 0) {
                for (auto& pair : weights1) {
                    pair.second /= total_weight1;
                }
            }
            
            if (total_weight2 > 0) {
                for (auto& pair : weights2) {
                    pair.second /= total_weight2;
                }
            }
            
            // Calculate weighted Jaccard
            double intersection_sum = 0.0;
            double union_sum = 0.0;
            
            std::unordered_set<int> all_keys;
            for (const auto& pair : weights1) all_keys.insert(pair.first);
            for (const auto& pair : weights2) all_keys.insert(pair.first);
            
            for (int key : all_keys) {
                double w1 = weights1.count(key) ? weights1[key] : 0.0;
                double w2 = weights2.count(key) ? weights2[key] : 0.0;
                intersection_sum += std::min(w1, w2);
                union_sum += std::max(w1, w2);
            }
            
            if (union_sum > 0) {
                double jaccard = intersection_sum / union_sum;
                log_message("jaccard_composite: Calculated weighted jaccard = " + std::to_string(jaccard));
                return jaccard;
            } else {
                log_message("jaccard_composite: Union sum is 0, returning 0");
                return 0.0;
            }
        }
        else {
            log_message("jaccard_composite: Unknown jaccard_type: " + jaccard_type + ", returning 0");
            return 0.0;
        }
    }
    catch (const std::exception& e) {
        log_message("ERROR in jaccard_composite: " + std::string(e.what()));
        return 0.0;
    }
    catch (...) {
        log_message("UNKNOWN ERROR in jaccard_composite");
        return 0.0;
    }
}