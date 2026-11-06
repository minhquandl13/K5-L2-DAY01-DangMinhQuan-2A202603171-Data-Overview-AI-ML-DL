# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

Ngày chạy: 06/11/2026

Runtime Colab: GPU T4

Python / PyTorch / Ultralytics: Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

Checkpoint: `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

Ngưỡng detection/segmentation dùng khi tạo JSON: `0.35`

Thay đổi so với notebook nguồn: Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   Record này mô tả nội dung tổng thể của cả bức ảnh, không xác định vị trí của một vật thể riêng biệt. Model Classification phân tích hình ảnh và đưa ra một lớp đại diện cho nội dung ảnh. Trong trường hợp này, model dự đoán lớp cab (taxi) có xác suất dự đoán cao nhất.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   Danh sách tên lớp được định nghĩa sẵn trong taxonomy đi kèm checkpoint. Ở bài thực hành này, model sử dụng taxonomy ImageNet-1K, gồm 1.000 lớp đã được định nghĩa. Model không tự tạo ra tên lớp mới mà lựa chọn lớp phù hợp nhất trong danh sách có sẵn.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   Cần lưu cả ba thông tin để xác định chính xác lớp được dự đoán:
   class_id: Mã số định danh của lớp, thuận tiện cho việc xử lý dữ liệu bằng chương trình.
   class_name: Tên lớp giúp con người dễ đọc và hiểu kết quả.
   taxonomy_name: Cho biết hệ thống phân loại đang sử dụng, giúp tránh nhầm lẫn giữa các bộ taxonomy khác nhau.
   Ví dụ, class_id = 468 và class_name = cab thuộc taxonomy ImageNet-1K.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?
   Guideline cần quy định rõ tiêu chí xác định lớp đại diện cho toàn bộ ảnh. Ví dụ: ưu tiên chủ thể chính, vật thể nổi bật nhất hoặc đối tượng chiếm vai trò quan trọng nhất trong ảnh. Những trường hợp có nhiều chủ thể tương đương cũng cần có quy tắc thống nhất để các annotator không đưa ra kết quả khác nhau.
6. Vì sao không được lấy `score` của model làm ground truth?
   Vì score chỉ thể hiện mức độ tin cậy của model đối với dự đoán, không chứng minh dự đoán đó là chính xác. Trong ví dụ này, model dự đoán cab với score 0.510915, nhưng đây vẫn chỉ là prediction. Ground truth phải được xác định thông qua việc con người kiểm tra nội dung ảnh và áp dụng guideline đã thống nhất.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= `person` ; `score`= 0.8734 ; `bbox_xyxy`= [238.0, 72.0, 471.0, 426.0] ; `bbox_width`= 471 - 238 = 233 px ; `bbox_height`= 426 - 72 = 354 px
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

- Góc trên bên trái nằm tại pixel (238, 72) — cách mép trái ảnh 238 px và cách mép trên ảnh 72 px.
- Góc dưới bên phải nằm tại (471, 426) — cách mép trái 471 px và cách mép trên 426 px (gần sát mép dưới của ảnh).
- Bounding box có chiều rộng 233 px và chiều cao 354 px.
  Đây là một bounding box cao và hẹp, bao quanh người đang đứng, kéo dài từ gần đường trần nhà xuống khu vực quầy bếp/sàn nhà.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   Ví dụ về số lượng trong một ô notebook thông thường — hãy thay thế bằng số liệu thực tế của bạn:

- ≥ 0.20 → khoảng 11 dự đoán (các bounding box thuộc đối tượng trong ảnh nhà bếp có điểm số trên 0.2 đều được giữ lại).
- ≥ 0.35 → khoảng 6 dự đoán.
- ≥ 0.60 → khoảng 3 dự đoán.
  Xu hướng: Khi giảm ngưỡng từ 0.60 xuống 0.20, số lượng bounding box được giữ lại tăng khoảng 2–3 lần.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

- Kiểm tra lại từng bounding box mới được bổ sung có độ tin cậy thấp: đó là đối tượng thực sự, bounding box trùng lặp/chồng lấn trên cùng một đối tượng hay dự đoán sai (False Positive), chẳng hạn như bóng đổ, hoa văn hoặc nhãn.
- Loại bỏ các bounding box trùng lặp hoặc chồng lấn trên cùng một đối tượng.
- Xác minh lại nhãn lớp (Class Label) có chính xác hay không. Điểm tin cậy thấp thường cho thấy mô hình đang phân vân giữa nhiều lớp đối tượng.
- Kiểm tra các bounding box yếu có rơi vào những đối tượng bị che khuất (Occluded) hoặc bị cắt khỏi khung hình (Truncated) hay không. Những trường hợp này có thể cần được đánh dấu bỏ qua hoặc xử lý riêng.
- Dành thêm thời gian kiểm tra các dự đoán sai (False Positive). Càng có nhiều bounding box thì càng có nhiều khả năng người đánh giá vô tình xác nhận một dự đoán sai.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

- Vẽ một bounding box hình chữ nhật, thẳng theo trục tọa độ (Axis-aligned), ôm sát một đối tượng riêng lẻ và bao phủ toàn bộ phần đối tượng có thể nhìn thấy.
- Bounding box phải chạm đến giới hạn ngoài cùng của đối tượng ở cả bốn phía — không để khoảng trống dư thừa (Padding) và không được cắt mất những phần còn nhìn thấy của đối tượng.
- Mỗi đối tượng chỉ có một bounding box; không gộp nhiều đối tượng vào cùng một bounding box.
- Nếu đối tượng bị che khuất đáng kể hoặc bị cắt bởi mép ảnh, chỉ vẽ bounding box quanh phần đối tượng còn nhìn thấy và gắn nhãn truncated/occluded khi phần nhìn thấy ít hơn khoảng 50%.
- Chỉ gán nhãn những đối tượng mà bạn có đủ cơ sở để nhận diện chính xác. Bỏ qua những đối tượng còn mơ hồ và ghi chú để người khác kiểm tra lại thay vì tự phỏng đoán.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   Ví dụ:

- Một chai nhỏ trên quầy bếp bị người đứng phía trước che khuất một phần.
- Một chiếc ghế bị cắt mất một phần do nằm sát mép phải của ảnh.
  Tôi sẽ đưa trường hợp này vào quy tắc hướng dẫn bằng văn bản thay vì chuyển lên cấp cao hơn để xử lý (Escalate).

Lý do: Các tình huống bị cắt ở mép ảnh hoặc bị che khuất một phần là những trường hợp thường xuyên và có thể dự đoán trước. Chúng sẽ xuất hiện trong hầu hết các bộ ảnh, vì vậy cần có một tiêu chuẩn xử lý thống nhất bằng văn bản.

Ví dụ về quy tắc:

- Gán nhãn đối tượng nếu phần nhìn thấy chiếm từ 50% trở lên.
- Nếu phần nhìn thấy ít hơn 50%, đánh dấu occluded hoặc bỏ qua theo quy định của dự án.
  Chỉ chuyển vấn đề lên Lab Coach khi gặp một trường hợp cụ thể, hiếm gặp mà quy tắc hiện tại không thể giải quyết.

Ví dụ: Một vùng hình ảnh bị mờ, không thể xác định đó là chai nước hay vòi nước.
Đây là những trường hợp cần đánh giá riêng lẻ, không phải vấn đề thiếu sót trong chính sách gán nhãn.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-001; `class_name`= person; `score`= `0.8993` ; số điểm polygon=`polygon_point_count`= 348 ; 3–5 điểm đầu của `polygon_xy`= `(446.7, 70.0)`, `(445.7, 71.0)`, `(444.7, 71.0)`, `(443.7, 72.0)`, `(442.7, 72.0)` ; bbox_xyxy = [385.45, 66.44, 498.02, 348.58]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   Bounding Box chỉ xác định vùng hình chữ nhật chứa đối tượng, trong khi Polygon mô tả đường viền thực tế của đối tượng bằng nhiều điểm tọa độ (ở đây là 348 điểm).
   Polygon giúp xác định chính xác hình dạng, ranh giới, các vùng lõm và phân biệt phần đối tượng với nền ảnh, ví dụ khoảng trống giữa cánh tay và thân người. Nhờ đó, có thể đánh giá độ chính xác của mask ở cấp độ pixel thay vì chỉ dựa trên độ chồng lấn của Bounding Box.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   instance_id là mã định danh riêng cho từng đối tượng trong một ảnh, giúp phân biệt các cá thể thuộc cùng một class.

- Không phải Class ID: Hai đối tượng person có thể có cùng class nhưng khác instance_id, ví dụ kitchen-001 và kitchen-009.
- Không phải Tracking ID: Mã này chỉ có giá trị trong một ảnh, không dùng để theo dõi đối tượng qua nhiều khung hình hoặc video.
  Mục đích chính là giúp nhận diện từng cá thể riêng biệt trong quá trình review, kiểm tra chất lượng (QA) và xử lý các trường hợp gán nhãn trùng lặp hoặc tranh chấp.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   Vẽ Polygon sát đường viền ngoài của phần đối tượng nhìn thấy, không bao gồm nền, bóng đổ hoặc vật thể bên cạnh. Mỗi đối tượng sử dụng một Polygon khép kín riêng, không gộp nhiều đối tượng vào cùng một Mask. Các điểm cần đủ chi tiết để thể hiện góc và vùng lõm, tránh tạo quá nhiều điểm không cần thiết hoặc vẽ cắt ngang các bộ phận của đối tượng. Với vật thể bị che khuất hoặc cắt khỏi mép ảnh, chỉ vẽ phần nhìn thấy và đánh dấu occluded hoặc truncated. Nếu ranh giới không rõ ràng, cần ghi chú để kiểm tra thay vì tự phỏng đoán.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   Ví dụ: kitchen-005 – dining table (score = 0.578, polygon có 598 điểm).

Mặt bàn có diện tích lớn, đồng thời tiếp xúc với người và các chiếc bát đặt trên bàn. Đây là vùng dễ nhầm lẫn về ranh giới giữa các đối tượng.

Khi gán nhãn, cần xác định chính xác đường viền của mặt bàn và từng chiếc bát, không để Mask của chúng bị gộp vào nhau. Nếu ranh giới bị che khuất hoặc không rõ, cần ghi chú để reviewer kiểm tra lại.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                        | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                     | Annotator làm gì?                                                                         | Reviewer xem gì?                                                                                      |
| --------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Một nhãn lớp cho mỗi ảnh, ví dụ `kitchen`.                           | Ảnh có nhiều vật thể thuộc các lớp khác nhau, dễ nhầm giữa nhãn chính và nhãn phụ.                                                     | Xác định nội dung chính và gán nhãn theo guideline.                                       | Kiểm tra nhãn có đúng với nội dung ảnh và thống nhất với guideline không.                             |
| Phát hiện vật thể     | Mỗi vật thể có một nhãn lớp và một Bounding Box (`bbox_xyxy`).       | Các box có thể chồng lấn, nhận diện sai hoặc bỏ sót vật thể. Cần kiểm tra prediction có confidence thấp trong JSON/PNG.                | Vẽ box sát vật thể, gán đúng class và xử lý các trường hợp bị che khuất hoặc cắt mép.     | Kiểm tra vị trí box, class label, vật thể bị bỏ sót và False Positive.                                |
| Instance segmentation | Mỗi cá thể có một nhãn lớp, `instance_id` và một Polygon/Mask riêng. | Ranh giới khó xác định khi vật thể tiếp xúc hoặc bị che khuất. Ví dụ `kitchen-005` (dining table) tiếp xúc với các chiếc bát trên bàn. | Vẽ Polygon theo đường viền phần nhìn thấy, không gộp các cá thể và ghi chú vùng không rõ. | Kiểm tra độ chính xác của Mask, ranh giới giữa các vật thể và các trường hợp gán nhãn sai hoặc thiếu. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   Không chia sẻ, sao chép hoặc tải dữ liệu lab lên các nền tảng bên ngoài khi chưa được cấp phép; chỉ sử dụng dữ liệu đúng mục đích và trong phạm vi được giao.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   Tôi sẽ dừng xử lý, không tự ý chỉnh sửa hoặc chia sẻ dữ liệu và báo ngay cho Lab Coach hoặc người phụ trách dự án để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
