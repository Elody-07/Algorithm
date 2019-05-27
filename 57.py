def FindNumbersWithSum(arr, sum):
    if len(arr) == 0:
        return None 
    
    start = 0
    end = len(arr) - 1
    while start < end :
        curSum = arr[start] + arr[end]
        if curSum == sum:
            return (start, end)
        elif curSum < sum:
            start += 1
        else:
            end -= 1
    return False # 找不到

# print(FindNumbersWithSum([1,2,4,7,11,15], 16))
# print(FindNumbersWithSum([1,2,4,7,11,15], 13))
# print(FindNumbersWithSum([1,2,3,4,11,15], 20))
# print(FindNumbersWithSum([], 5))

def FindContinuousSeq(sum):
    if sum < 3:
        return False
    
    small = 1
    big = 2
    mid = (sum - 1) // 2
    curSum = small + big
    res = []

    while small <= mid :
        if curSum == sum:
            res += [list(range(small, big+1))]
        
        while curSum > sum and small <= mid :
            curSum -= small
            small += 1

            if curSum == sum:
                res += [list(range(small, big+1))]
        
        big += 1
        curSum += big
    return res

print(FindContinuousSeq(15))
print(FindContinuousSeq(4))