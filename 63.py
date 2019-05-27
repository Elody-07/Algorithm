def MaxDiff(arr):
    if len(arr) < 2:
        return False
    
    min = arr[0]
    maxDiff = arr[1] - min

    for i in range(2, len(arr)):
        if arr[i - 1] < min:
            min = arr[i - 1]
        
        curDiff = arr[i] - min
        if curDiff > maxDiff:
            maxDiff = curDiff
    
    return maxDiff

print(MaxDiff([]))
print(MaxDiff([9, 11, 8, 5, 7, 12, 16, 14]))
print(MaxDiff([9, 9]))
print(MaxDiff([9, 10, 11]))
print(MaxDiff([9, 8, 7]))