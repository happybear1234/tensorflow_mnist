def bubble_sort(lis):
    for item in range(len(lis) - 1, 0, -1):
        for i in range(0, item):
            if lis[i] > lis[i + 1]:
                temp = lis[i + 1]
                lis[i + 1] = lis[i]
                lis[i] = temp
    return lis

if __name__ == '__main__':
    lis = [3, 0, 5, 0, 1, 2, 3, 9, 8]
    list=bubble_sort(lis)
    print(lis)
