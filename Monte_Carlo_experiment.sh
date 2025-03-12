#!/bin/bash

# 定义要测试的信噪比列表
snr_values=(-40 -38 -36 -34 -32 -30 -28 -26 -24 -22 -20 -18 -16 -14 -12 -10 -8 -4 -2 0)
output_file="results.txt"

# CA算法
echo "CA算法实验-----------------------------------------" >> $output_file
for snr in "${snr_values[@]}" # 遍历每个信噪比
do
    echo "正在运行信噪比为 $snr 的实验..."
    # 执行 Python 脚本并将结果追加到结果文件中
    python CA_cupy.py --snr $snr -o $output_file
done

# GO算法
echo "GO算法实验-----------------------------------------" >> $output_file
for snr in "${snr_values[@]}"
do
    echo "正在运行信噪比为 $snr 的实验..."
    # 执行 Python 脚本并将结果追加到结果文件中
    python GO_cupy.py --snr $snr -o $output_file
done

#SO算法
echo "SO算法实验-----------------------------------------" >> $output_file
for snr in "${snr_values[@]}"
do
    echo "正在运行信噪比为 $snr 的实验..."
    # 执行 Python 脚本并将结果追加到结果文件中
    python SO_cupy.py --snr $snr -o $output_file
done

#OS算法
echo "OS算法实验-------------------------------------------" >> $output_file
for snr in "${snr_values[@]}"
do
    echo "正在运行信噪比为 $snr 的实验..."
    # 执行 Python 脚本并将结果追加到结果文件中
    python OS_cupy.py --snr $snr -o $output_file
done

echo "所有实验完成，结果已保存到 $output_file"