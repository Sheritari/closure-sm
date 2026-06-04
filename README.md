# Перенос на Linux

## Содержимое папки

| Файл | Назначение |
|------|------------|
| `closure_sm_batch.cu` | исходник |
| `k5_ge36000_orbit60.txt` | список мультиопераций |
| `build.sh` | сборка через `nvcc` |

## Сборка

```bash
cd closure_sm_batch_linux
chmod +x build.sh
./build.sh
```

```bash
./closure_sm_batch --orbit60 k5_ge36000_orbit60.txt --parallel 8 --no-mu-lut
```

## Запуск

```bash
./closure_sm_batch --orbit60 k5_ge36000_orbit60.txt --parallel 8 2>&1 | tee orbit60_run.log
```
