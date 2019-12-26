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


class Solution(object):
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

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if s == '':
            return 0
        
        position = [-1] * 256
        cur = 0
        longest = 0
        for i in range(len(s)):
            prevIdx = position[ord(s[i])]
            if prevIdx == -1 or (i - prevIdx) > cur:
                cur += 1
            elif i - prevIdx < cur:
                cur = i - prevIdx
            if cur > longest:
                longest = cur
            position[ord(s[i])] = i
        return longest

class Solution:
    def reverse(self, x: int) -> int:
        if x < -2**31 or x > (2**31 - 1):
            return 0
        
        flag = 1
        if x < 0:
            flag = -1
        x *= flag
        nums = []
        while(x >= 1):
            nums.append(int(x % 10))
            x /= 10.

        reverse = 0
        for idx, num in enumerate(nums):
            reverse += num * (10**(len(nums) -idx - 1))
        return flag * reverse

class Solution:
    def longestPalindrome(self, s: str) -> str:
        longest = ''
        for i in range(len(s)):
            for j in range(len(s)-1, i-1, -1):
                if s[i] == s[j] and self.isPalindrome(s[i:j+1]) and (j-i+1) > len(longest):
                    longest = s[i: j+1]
        return longest
    
    
    def isPalindrome(self, s: str) -> bool:
        if s == '':
            return False
        
        res = True
        for i in range(len(s) // 2):
            if s[i] != s[len(s)-i-1]:
                res = False
                break
        return res
        
        
class Solution:
    def removeDuplicates(self, nums: list) -> int:
        if nums == []:
            return 0
        
        slow = 0
        fast = 0
        while(fast < len(nums)):
            while fast < len(nums) - 1 and nums[fast] == nums[fast+1]:
                fast += 1
            nums[slow] = nums[fast]
            slow += 1
            fast += 1
        
        return nums[:slow]
                

class Solution:
    def countAndSay(self, n: int) -> str:
        if n <= 0:
            return ''
        if n == 1:
            return '1'

        
        nums = [1]
        for i in range(1, n):
            nums = self.countNums(nums)
        
        return ''.join(str(x) for x in nums)
    
    def countNums(self, nums: list) -> list:
        count = 1
        res = []
        for i in range(len(nums)):
            if i == len(nums) - 1 or nums[i] != nums[i+1]:
                res.append(count)
                res.append(nums[i])
                count = 1
            else:
                count += 1
        return res

class Solution:
    def merge(self, nums1, m: int, nums2, n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        length = m + n - 1
        idx1 = m - 1
        idx2 = n - 1
        
        while(idx1 >= 0 and idx2 >= 0 and length >= 0):
            if nums1[idx1] > nums2[idx2]:
                nums1[length] = nums1[idx1]
                idx1 -= 1
            else:
                nums1[length] = nums2[idx2]
                idx2 -= 1
            length -= 1
        
        if idx2 >= 0 and length >= 0:
            nums1[:length+1] = nums2[:idx2+1]
        return nums1

        
        
ans = Solution()
print(ans.merge([0],0,[1],1))
print('Done')