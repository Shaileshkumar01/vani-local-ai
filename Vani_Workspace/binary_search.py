def binary_search(arr: list[int], target: int) -> int:
    """Return index of target in sorted list, or -1 if not found."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


if __name__ == "__main__":
    nums = [2, 5, 8, 12, 16, 23, 38, 56, 72]
    print("Array:", nums)
    print("Index of 23:", binary_search(nums, 23))
