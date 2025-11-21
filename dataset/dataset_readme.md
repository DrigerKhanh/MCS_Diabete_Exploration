This dataset contains 3 files:

- diabetes _ 012 _ health _ indicators _ BRFSS2015.csv is a clean dataset of 253,680 survey responses to the CDC's 
BRFSS2015. The target variable Diabetes_012 has 3 classes. 0 is for no diabetes or only during pregnancy, 1 is for 
prediabetes, and 2 is for diabetes. There is class imbalance in this dataset. This dataset has 21 feature variables

- diabetes _ binary _ 5050split _ health _ indicators _ BRFSS2015.csv is a clean dataset of 70,692 survey responses to 
the CDC's BRFSS2015. It has an equal 50-50 split of respondents with no diabetes and with either prediabetes or diabetes. 
The target variable Diabetes_binary has 2 classes. 0 is for no diabetes, and 1 is for prediabetes or diabetes. This dataset 
has 21 feature variables and is balanced.

- diabetes _ binary _ health _ indicators _ BRFSS2015.csv is a clean dataset of 253,680 survey responses to the CDC's 
BRFSS2015. The target variable Diabetes_binary has 2 classes. 0 is for no diabetes, and 1 is for prediabetes or diabetes. 
This dataset has 21 feature variables and is not balanced.


HEATMAP
"Heatmap giúp em trả lời 3 câu hỏi quan trọng:
- Biến nào ảnh hưởng nhất đến bệnh tiểu đường? → Nhìn cột Diabetes_binary
- Có vấn đề multicollinearity không? → Tìm các ô đỏ ngoài đường chéo
- Cấu trúc dataset thế nào? → Các cụm biến tương quan với nhau

Đặc điểm 📊 THANG ĐO TƯƠNG QUAN:
Giá trị	    Mức độ	            Ý nghĩa
0.7 - 1.0	Rất mạnh	    Hai biến gần như thay thế được nhau
0.5 - 0.7	Mạnh	        Liên quan chặt chẽ
0.3 - 0.5	Trung bình	    Có mối liên hệ rõ ràng
0.1 - 0.3	Yếu	            Liên quan nhẹ
0.0 - 0.1	Rất yếu	        Hầu như không liên quan

CÂU HỎI QUAN TRỌNG BIỂU ĐỒ NÀY GIÚP TRẢ LỜI:
Với nghiên cứu tiểu đường:
"Dân số trong dataset có đại diện không?" (tuổi, BMI)
"Tình trạng sức khỏe chung của population?" (GenHlth)
"Có vấn đề sức khỏe mãn tính phổ biến không?" (PhysHlth)

APRIORI ALGORITHM
- https://www.geeksforgeeks.org/machine-learning/apriori-algorithm/