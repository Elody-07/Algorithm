def QuickSort(arr, start, end):
    if len(arr) == 0:
        return False 
    
    index = Partition(arr, start, end)
    if index > start:
        QuickSort(arr, start, index - 1)
    if index < end:
        QuickSort(arr, index + 1, end)
    return arr

def Partition(arr, start, end):
    assert start <= end
    small = start - 1

    for i in range(start, end):
        if arr[i] < arr[end]:
            small += 1
            if small != i:
                temp = arr[small]
                arr[small] = arr[i]
                arr[i] = temp
    
    small += 1
    temp = arr[small]
    arr[small] = arr[end]
    arr[end] = temp
    return small

l = [3,2,1,5,8,5]
print(QuickSort(l, 0, len(l)-1))
