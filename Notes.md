# 1. Minimum Support = 0.01 có quá thấp?

`min_support = 0.01` **không hề quá thấp**, mà thực tế **là một lựa chọn tốt và phù hợp** cho bài toán này.  
Dưới đây là các lập luận cụ thể:

---

## a. Bằng chứng từ kết quả đầu ra

- Các luật thu được có **Support phổ biến trong khoảng 0.03 đến 0.1** (tức **3% đến 10%**).  
- **Rule phổ biến nhất** (Rule 2819): `Support = 0.107` (10.7%)  
- **Rule ít phổ biến nhất** trong top 10 (Rule 5013): `Support = 0.031` (3.1%)

👉 Điều này chứng tỏ:
- Thuật toán đã **lọc bỏ** rất nhiều các itemset có Support thấp hơn (từ 0.01 đến 0.03) để chỉ giữ lại những luật mạnh.  
- Nếu `min_support = 0.01` là quá thấp, ta đã thấy nhiều luật yếu (0.01, 0.012, …) trong top 10 — nhưng thực tế **không hề có**.

---

## b. Bản chất của các yếu tố nguy cơ y tế

Các tổ hợp như:

> `[HeartDiseaseorAttack, BMI_VeryHigh, PhysHealth_Poor]`

vốn **không phổ biến trong cộng đồng chung**, mà thuộc về **nhóm bệnh nhân nguy cơ cao**.  

- Nếu chọn ngưỡng `support` cao hơn (ví dụ `0.1` hay 10%),  
  → các nhóm nhỏ nhưng **nguy cơ cao** này sẽ bị loại bỏ.  
- Trong y tế, việc phát hiện **nhóm nhỏ – rủi ro lớn** lại có **ý nghĩa cực kỳ quan trọng**  
  → giúp đưa ra các biện pháp can thiệp mạnh và chính xác.

---

## c. Cân bằng giữa "Độ phổ biến" và "Tính đặc thù"

Lựa chọn `min_support = 0.01` mang lại sự **cân bằng hợp lý**:

✅ Giữ lại các **mẫu quan trọng, có ý nghĩa lâm sàng**, dù chúng không phổ biến.  
✅ Loại bỏ **hàng nghìn tổ hợp cực hiếm**, vô nghĩa về mặt thống kê (`Support < 0.01`).  
✅ Giúp **thuật toán chạy nhanh hơn** và **kết quả gọn gàng, dễ phân tích hơn**.

---

> 🔹 **Kết luận:**  
> `min_support = 0.01` là một **mức hợp lý**, vừa đảm bảo tính **bao quát dữ liệu**, vừa duy trì **ý nghĩa thực tiễn và hiệu năng xử lý**.
