# 定义要测试的信噪比列表
$snr_values = @(-10, -5, 0, 5, 10, 15, 20)

# 定义结果文件
$output_file = "result.txt"

# 清空结果文件（如果存在）
if (Test-Path $output_file) {
    Clear-Content $output_file
} else {
    New-Item -Path $output_file -ItemType File
}

# 遍历每个信噪比
foreach ($snr in $snr_values) {
    Write-Host "正在运行信噪比为 $snr 的实验..."
    
    # 执行 Python 脚本并将结果追加到结果文件中
    python one_pulse_only.py --snr $snr -o $output_file
    
    # 添加分隔线到结果文件
    "------------------------------------------------" | Add-Content $output_file
}

Write-Host "所有实验完成，结果已保存到 $output_file"