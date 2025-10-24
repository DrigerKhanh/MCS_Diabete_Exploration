Steps để chạy annaconda
- conda create --name diablet-exploration python=<version>: Tạo môi trường annaconda mới (3.12)
- conda activate diablet-exploration (hoặc env mà mình cần chạy) - kích hoạt môi trường <env> của annaconda
- conda deactivate diablet-exploration : Tắt môi trường annaconda
- conda env remove --name diablet-exploration : Xoá môi trường <env> annaconda
- conda env list : Liệt kê các môi trường

Requirement:
- pip install pandas, numpy, matplotlib, seaborn, scipy