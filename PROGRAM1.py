n=int(input("Enter the number of elements:"))
data=[]
print("Enter the elements:")
for i in range(n):
      num=float(input())
      data.append(num)
total=0
for i in data:
      total+=i
mean=total/n

for i in range(n):
      for j in range(0,n-i-1):
          if data[j]>data[j+1]:
             temp=data[j]
             data[j]=data[j+1]
             data[j+1]=temp
if n%2==0:
    median=(data[n//2-1]+data[n//2])/2
else:
    median=data[n//2]

max_count=0
mode=None

for i in range(n):
    count=0
    for j in range(n):
        if data[i]==data[j]:
            count+=1
    if count>max_count:
        max_count=count
        mode=data[i]
    if max_count==1:
        mode="No Mode"

sum_sq=0
for i in data:
    sum_sq+=(i-mean)**2
variance=sum_sq/n

guess=variance

if variance==0:
    std_dev=0
else:
    for i in range(20):
        guess=(guess+variance/guess)/2
    std_dev=guess

print("\n Results")
print("----------")
print("Mean=",mean)
print("Median=",median)
print("Mode=",mode)
print("Variance=",variance)
print("Standard Deviation=",std_dev)
          
