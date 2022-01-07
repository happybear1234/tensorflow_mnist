import pandas as pd


# pd.set_option('display.max_columns',None)
# pd.set_option('display.max_rows',None)

def read_xls_2020(file):
    data = pd.read_excel(file, na_values='/')
    print(data.iloc[0])
    data = data[2:8]
    row, col = data.shape
    print(data.iloc[0])

    r1, r1_1, r2, r2_1, r3, r3_1, r4, r4_1, r5 = [], [], [], [], [], [], [], [], []
    for i in range(row):
        # for j in range(col):
        result_1 = data.iloc[i, 3]  # 笔试
        result_1_1 = result_1 * 0.3
        result_2 = data.iloc[i, 4]  # 试讲
        result_2_1 = result_2 * 0.15
        result_3 = data.iloc[i, 5]  # 技能
        result_3_1 = result_3 * 0.15
        result_4 = data.iloc[i, 7]  # 面试
        result_4_1 = result_4 * 0.4
        result_5 = data.iloc[i, 8]  # 总分
        print('笔试：{:.1f}*0.3={:.2f},试讲：{:.1f}*0.15={:.2f}，技能：{:.1f}*0.15={:.2f}，面试：{:.1f}*0.4={:.2f}，总分:{}|{}'
              .format(result_1, result_1_1, result_2, result_2_1, result_3, result_3_1, result_4, result_4_1,
                      result_1_1 + result_2_1 + result_3_1 + result_4_1, result_5))
        r1.append(result_1)
        r2.append(result_2)
        r3.append(result_3)
        r4.append(result_4)
        r5.append(result_5)
        r1_1.append(result_1_1)
        r2_1.append(result_2_1)
        r3_1.append(result_3_1)
        r4_1.append(result_4_1)
    datas={'笔试':r1,'笔试折算':r1_1,'试讲':r2,'试讲折算':r2_1,'技能':r3,'技能折算':r3_1,'面试':r4,'面试折算':r4_1,'总分':r5}
    datas=pd.DataFrame(datas)
    return datas

def to_excel(file, datas):
    datas.to_excel(file,index=False)


if __name__ == '__main__':
    datas=read_xls_2020('./result_2020.xls')
    print(datas)
    to_excel('./input.xls',datas)
