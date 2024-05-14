#include <iostream>
#include <vector>

using namespace std;

// Binary search to find the lower bound
int lower_bound_binary_search(const vector<int>& arr, int x) {
    int low = 0, high = arr.size() - 1, result = -1;

    while (low <= high) {
        int mid = low + (high - low) / 2;

        if (arr[mid] <= x) {
            result = mid;  // Update result and continue searching in the right half
            low = mid + 1;
        } else {
            high = mid - 1;  // Search in the left half
        }
    }

    return result;
}

int main() {
    // Example usage
    vector<int> arr = {1, 2, 4, 4, 6, 6, 7, 8, 9};
    int x = -1;

    int lowerBoundIndex = lower_bound_binary_search(arr, x);

    if (lowerBoundIndex != -1) {
        cout << "Lower bound of " << x << " is at index: " << lowerBoundIndex << endl;
        cout << "Value at lower bound index: " << arr[lowerBoundIndex] << endl;
    } else {
        cout << "No lower bound found for " << x << " in the vector." << endl;
    }

    return 0;
}
