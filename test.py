import random

nums = [random.randint(1, 100) for _ in range(20)]

# ============================= INSERTION SORT =============================
def insertion_sort(nums):
    n = len(nums)
    for i in range(1, n):
        while i > 0 and nums[i] < nums[i-1]:
            nums[i], nums[i-1] = nums[i-1], nums[i]
            i= i-1

    return(nums)

print(f"nums: {nums}")
print(f"Insertion_sort_nums: {insertion_sort(nums)}")

# ============================= MERGE SORT =============================
def merge_sort(nums):
    def merge(A, B):
        comb = []
        a = b = 0
        
        while a < len(A) and b < len(B):
            if A[a] <= B[b]:
                comb.append(A[a])
                a += 1
            else:
                comb.append(B[b])
                b += 1
        
        while a < len(A):
            comb.append(A[a])
            a += 1
        
        while b < len(B):
            comb.append(B[b])
            b += 1
        
        return comb

    def merge_sort_helper(list):
        n = len(list)
        if n == 1:
            return list
        
        mid = len(list) // 2
        left = list[:mid]
        right = list[mid:]
        
        return merge(merge_sort_helper(left), merge_sort_helper(right))
    
    return merge_sort_helper(nums)

nums = [random.randint(1, 100) for _ in range(20)]

print(f"nums: {nums}")
print(f"Merge_sort_nums: {insertion_sort(nums)}")

# ============================= QUICK SORT =============================
def quick_sort(nums):
    def pivot(nums, left, right):
        if right <= left:
            return left
        
        swap = pivot = left
        for i in range(left + 1, right):
            if nums[i] < nums[pivot]:
                swap += 1
                nums[swap], nums[i]= nums[i], nums[swap]
        
        nums[swap], nums[pivot] = nums[pivot], nums[swap]
        return swap
    
    def quick_sort_helper(nums, left, right):
        if left < right:
            pivot_idx = pivot(nums, left, right)
            quick_sort_helper(nums, left, pivot_idx)
            quick_sort_helper(nums, pivot_idx+1, right)
        
        return nums
    
    return quick_sort_helper(nums, 0, len(nums))

nums = [random.randint(1, 100) for _ in range(20)]

print(f"nums: {nums}")
print(f"Quick_sort_nums: {insertion_sort(nums)}")

# ============================= DIJKSTRAS ALGORITHM =============================

n = 5
graph = [[] for _ in range(n)]

graph[0].append((1, 2))
graph[0].append((2, 4))
graph[1].append((2, 1))
graph[1].append((3, 7))
graph[2].append((4, 3))
graph[3].append((4, 1))

import heapq

def dijkstra(n, graph, start):
    distances = [float('inf') for _ in range(n)]
    
    pq = [(0, start)]
    distances[start] = 0
    
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        for neigh_node, weight in graph[current_node]:
            dist = current_dist + weight
            
            if dist > distances[neigh_node]:
                continue
            else:
                distances[neigh_node] = dist
                heapq.heappush(pq, (dist, neigh_node))
    
    return distances

print(dijkstra(n, graph, 0))

# ============================= MAXIMUM PRODUCT SUBARRAY (KADANES ALGORITHM) =============================

def max_product_subarray(arr):
    max_so_far = arr[0]
    maximum = arr[0]
    minimum = arr[0]
    
    for i in range(1, len(arr)):
        curr = arr[i]
        
        temp_maximum = max(
            curr,
            curr * maximum,
            curr * minimum
        )
        
        temp_minimum = min(
            curr,
            curr * maximum,
            curr * minimum
        )
        
        maximum = temp_maximum
        minimum = temp_minimum
        
        max_so_far = max(max_so_far, maximum)
    
    return max_so_far

nums = [2, 3, -2, 0, 4, 0, 5, -6, 1]

print(max_product_subarray(nums))

print(isinstance(max_product_subarray, type))
print(max_product_subarray.__class__)

class MyClass:
    def hello(self):
        print("Hello World!")

print(isinstance(MyClass, type))
print(MyClass.__class__)
print(type(MyClass))

myclass = MyClass()

print(isinstance(myclass, type))
print(isinstance(myclass, MyClass))
print(myclass.__class__)
print(type(myclass))

print(myclass.hello())
print('-------------------------------------------')
# print(MyClass.hello())
print('-------------------------------------------')
myclass.hello()
print('-------------------------------------------')

class YourClass:
    def hello():
        print("Hello World!")

yourclass = YourClass()

print(f"YourClass - {YourClass.hello()}")
# print(yourclass.hello())


import time

def tictoc(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"{func.__name__} ran in {t2 - t1} seconds!")
        return result
    return wrapper

@tictoc
def do_sleep(n):
    print(f"Staring sleep for {n} seconds.")
    time.sleep(n)
    print("Sleep done!")

do_sleep(2.3)

print(myclass.__dict__)
print(MyClass.__dict__)

print('-------------------------------------------')

class Person:
    def __init__(self, name):
        self.name = name

class Student(Person):
    def say_hello(self):
        return f"{self.name} says hello!"

s1 = Student("cherry")
print(s1.say_hello())
print(Student.__mro__)
