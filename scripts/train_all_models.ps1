# Automated Multi-Model Training Script
# Runs multiple models sequentially with proper output separation

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Multi-Model Training Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$python = ".\.venv\Scripts\python.exe"
$script = "scripts\train.py"

# Model configurations
$models = @(
    @{
        Name = "EfficientNet-B1"
        Config = "configs\efficientnet_b1.yaml"
        OutputDir = "outputs_efficientnet_b1"
    },
    @{
        Name = "ResNeXt-50"
        Config = "configs\resnext50.yaml"
        OutputDir = "outputs_resnext50"
    }
)

$startTime = Get-Date

foreach ($model in $models) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Training: $($model.Name)" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    
    $modelStartTime = Get-Date
    
    # Run training
    & $python $script --config $model.Config --output-dir $model.OutputDir
    
    $modelEndTime = Get-Date
    $modelDuration = $modelEndTime - $modelStartTime
    
    Write-Host ""
    Write-Host "[$($model.Name)] Completed in $($modelDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
    Write-Host ""
}

$endTime = Get-Date
$totalDuration = $endTime - $startTime

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  All Training Complete!" -ForegroundColor Cyan
Write-Host "  Total Time: $($totalDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor White
foreach ($model in $models) {
    Write-Host "  - $($model.OutputDir)/" -ForegroundColor Gray
}
