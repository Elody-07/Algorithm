class TreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

class Solution(object):
    def findTarget(self, root: TreeNode, target: int) -> bool:
        if root is None:
            return False
        
        midorder = []
        self.MidOrder(root, midorder)
        print(midorder)
        
        start = 0
        end = len(midorder) - 1
        curSum = midorder[start] + midorder[end]
        while(curSum != target and start < end):
            if curSum > target:
                end -= 1
            else:
                start += 1
            curSum = midorder[start] + midorder[end]
        if curSum == target:
            return True
        else:
            return False
    
    def MidOrder(self, root, res=[]):
        if root is None:
            return
        
        if root.left is not None:
            self.MidOrder(root.left, res)
            
        res.append(root.val)
        if root.right is not None:
            self.MidOrder(root.right, res)
            
        return

# a, b, c, d, e, f = TreeNode(5), TreeNode(3), TreeNode(6), TreeNode(2), TreeNode(4), TreeNode(7) 
# a.left, a.right, b.left, b.right, c.right = b, c, d, e, f

# ans = Solution()
# print(ans.findTarget(a, 28))
# print('Done')


class Solution:
    def reverse(self, x: int) -> int:
        nums = []
        flag = 1
        if x < 0:
            flag = -1
        x *= flag
        while x > 0:
            nums.append(int(x % 10))
            x = x // 10
        new = 0
        for i in range(len(nums)):
            new += flag * nums[i] * (10 ** (len(nums)-i-1))
        
        if new < (-1 * 2 ** 31) or new > (2**31 -1):
            new = 0
        
        return new


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) <= 0:
            return 0
        
        curLength = 0
        maxLength = 0
        position = {}
        
        for i in range(len(s)):
            if s[i] in position.keys():
                prevIdx = position[s[i]]
                if (i - prevIdx) > curLength:
                    curLength += 1
                else:
                    curLength = i - prevIdx
            else:
                curLength += 1
            
            if curLength > maxLength:
                maxLength = curLength
            
            position[s[i]] = i
        
        return maxLength
ans = Solution()
print(ans.lengthOfLongestSubstring(' abca'))
print('Done')