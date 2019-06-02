
def hasPath(arr, rows, cols, str):
    if arr == [] or rows < 1 or cols < 1 or str == '':
        return None
    
    visited = [0] * (rows * cols)
    for row in range(0, rows):
        for col in range(0, cols):
            if find(arr, rows, cols, row, col, str, visited):
                return True
    
    return False

def find(arr, rows, cols, row, col, str, visited):
    if str == '':
        return True
    
    flag = False
    if (row >= 0 and row < rows and 
        col >= 0 and col < cols and 
        arr[row*cols + col] == str[0] and 
        visited[row*cols + col] == 0):

        visited[row*cols + col] = 1

        flag = find(arr, rows, cols, row, col-1, str[1:], visited) or \
                  find(arr, rows, cols, row, col+1, str[1:], visited) or \
                  find(arr, rows, cols, row-1, col, str[1:], visited) or \
                  find(arr, rows, cols, row+1, col, str[1:], visited)
    
    return flag 

arr = ['a', 'b', 't', 'g',
       'c', 'f', 'c', 's',
       'j', 'd', 'e', 'h']
print(hasPath(arr, 3, 4, 'abfd'))
print(hasPath(arr, 3, 4, 'bfce'))
print(hasPath(arr, 3, 4, 'abfb'))
print(hasPath([], 0, 0, ''))
arr = ['a', 'a', 'a']
print(hasPath(arr, 3, 1, 'aaa'))
print(hasPath(arr, 3, 1, 'a'))
print(hasPath(arr, 3, 1, 'aaaa'))