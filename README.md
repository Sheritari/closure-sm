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

## Запуск

```bash
./closure_sm_batch 0x112698 --layers 20 2>&1 | tee orbit120_112698.log # orbit120
```