# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

Ngày chạy: 06/11/2026

Runtime Colab: GPU T4

Python / PyTorch / Ultralytics: Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

Checkpoint: `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

Ngưỡng detection/segmentation dùng khi tạo JSON: `0.35`

Thay đổi so với notebook nguồn: Không có thay đổi.

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
   Record này mô tả nội dung tổng thể của cả bức ảnh, không xác định vị trí của một vật thể riêng biệt. Model Classification phân tích hình ảnh và đưa ra một lớp đại diện cho nội dung ảnh. Trong trường hợp này, model dự đoán lớp cab (taxi) có Model Score cao nhất trong số các lớp được xét (lưu ý: score là điểm tin cậy của model, không nhất thiết là xác suất đã hiệu chuẩn).
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

1. Chép một record: `class_name`= `person` ; `score`= 0.912625 ; `bbox_xyxy`= [385.3, 69.2, 498.9, 348.9] ; `bbox_width`= 498.9 - 385.3 = 113.6 px ; `bbox_height`= 348.9 - 69.2 = 279.7 px
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

- Góc trên bên trái nằm tại pixel (385.3, 69.2) — cách mép trái ảnh 385.3 px và cách mép trên ảnh 69.2 px.
- Góc dưới bên phải nằm tại (498.9, 348.9) — cách mép trái 498.9 px và cách mép trên 348.9 px.
- Bounding box có chiều rộng 113.6 px và chiều cao 279.7 px.
  Đây là một bounding box cao và hẹp, bao quanh người đang đứng ở khu vực bên phải ảnh, kéo dài từ gần mép trên xuống khoảng 82% chiều cao ảnh.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   Số đếm thực tế trên ảnh `kitchen` (11 records trong JSON, JSON đã được lọc sẵn ở ngưỡng 0.35):

- ≥ 0.20 → 11 dự đoán (toàn bộ 11 box của ảnh kitchen).
- ≥ 0.35 → 11 dự đoán (vì file JSON đã được tạo tại ngưỡng 0.35 nên không có record nào nằm trong khoảng 0.20–0.35).
- ≥ 0.60 → 6 dự đoán.
  Xu hướng: Khi nâng ngưỡng từ 0.35 lên 0.60, số bounding box giảm từ 11 xuống 6 (loại bỏ các box điểm thấp: bowl, cup, person thứ hai...).

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
- Nếu đối tượng bị cắt bởi mép ảnh, vẽ bounding box quanh phần còn nhìn thấy và gắn nhãn `truncated`.
- Nếu đối tượng bị vật thể khác che khuất, vẽ bounding box quanh phần còn nhìn thấy và gắn nhãn `occluded`.
- Quyết định có bỏ qua đối tượng hay không, và ngưỡng phần trăm nhìn thấy tối thiểu (nếu có), phải tuân thủ theo guideline của dự án; không tự đặt ngưỡng thay vì guideline.
- Chỉ gán nhãn những đối tượng mà bạn có đủ cơ sở để nhận diện chính xác. Bỏ qua những đối tượng còn mơ hồ và ghi chú để người khác kiểm tra lại thay vì tự phỏng đoán.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

- Ví dụ có thật trong JSON: record `oven` score = 0.686791 có `bbox_xyxy` = [0.1, 187.5, 196.0, 292.8] — cạnh trái của box nằm sát tọa độ x = 0.1, tức là chiếc lò nướng bị cắt mất một phần bởi mép trái của ảnh (truncated).
- Ngoài ra còn có record `person` score = 0.610931 với box [0.1, 263.2, 61.5, 310.8] — một người nhỏ cũng bị cắt ở mép trái.

Tôi sẽ đưa trường hợp này vào quy tắc hướng dẫn bằng văn bản (guideline) thay vì chuyển lên Lab Coach (escalation), và xử lý theo nhãn `truncated` đã có trong guideline: chỉ vẽ quanh phần nhìn thấy của đối tượng, không vẽ phần bị mép ảnh cắt mất.

Lý do: các tình huống bị cắt ở mép ảnh (truncated) hoặc bị che khuất một phần là những trường hợp thường xuyên và có thể dự đoán trước; chúng xuất hiện trong hầu hết các bộ ảnh, nên cần có một tiêu chuẩn xử lý thống nhất bằng văn bản.

Cách xử lý các trường hợp này:

- Với đối tượng bị cắt mép hoặc bị che, gán nhãn phần nhìn thấy theo đúng guideline của dự án.
- Nếu không xác định được cách xử lý, hoặc guideline chưa quy định rõ trường hợp này, ghi chú lại và đưa lên Lab Coach để được hướng dẫn, không tự đặt ngưỡng 50% hay tự quyết định.

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
   Ví dụ: kitchen-005 – dining table (score = 0.5784, polygon có 598 điểm).

Mặt bàn có diện tích lớn, đồng thời tiếp xúc với người và các chiếc bát đặt trên bàn (các record kitchen-002, 003, 006, 010 là bowl). Đây là vùng dễ nhầm lẫn về ranh giới giữa mặt bàn và thành bát.

Quyết định: tôi dùng **guideline**, không escalation. Cụ thể, bổ sung một quy tắc vào guideline: khi vẽ mask, mỗi đối tượng là một polygon khép kín riêng; ranh giới giữa mặt bàn và chiếc bát là đường tiếp giáp nhìn thấy được — vẽ bám sát thành bát cho mask của bát, và mặt bàn chỉ kéo dài đến mép dưới của bát, không để mask hai đối tượng chồng lấn/gộp vào nhau.

Tuy nhiên, không phải lúc nào cũng nhận diện chính xác đường tiếp giáp. Nếu ranh giới giữa hai đối tượng không thể xác định rõ ràng từ ảnh, người gán nhãn cần đánh dấu vùng mơ hồ và chuyển cho reviewer hoặc Lab Coach đánh giá, không tự ý phỏng đoán. Chỉ **escalate** lên Lab Coach khi ranh giới thực sự quá mờ hoặc không thể xác định (ví dụ: bát bị che khuất hoàn toàn, không phân biệt được bát hay bóng trên mặt bàn).

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                        | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                     | Annotator làm gì?                                                                         | Reviewer xem gì?                                                                                      |
| --------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Một nhãn lớp cho mỗi ảnh, ví dụ `kitchen`.                           | Ảnh có nhiều vật thể thuộc các lớp khác nhau, dễ nhầm giữa nhãn chính và nhãn phụ.                                                     | Xác định nội dung chính và gán nhãn theo guideline.                                       | Kiểm tra nhãn có đúng với nội dung ảnh và thống nhất với guideline không.                             |
| Phát hiện vật thể     | Mỗi vật thể có một nhãn lớp và một Bounding Box (`bbox_xyxy`).       | Các box có thể chồng lấn hoặc bỏ sót vật thể; cần kiểm tra thêm các prediction có confidence thấp trong JSON/PNG — đây là điểm cần review, chưa khẳng định model sai. | Vẽ box sát vật thể, gán đúng class và xử lý các trường hợp bị che khuất hoặc cắt mép.     | Kiểm tra vị trí box, class label, vật thể bị bỏ sót và False Positive.                                |
| Instance segmentation | Mỗi cá thể có một nhãn lớp, `instance_id` và một Polygon/Mask riêng. | Ranh giới cần review khi vật thể tiếp xúc hoặc bị che, ví dụ `kitchen-005` (dining table) tiếp xúc với các chiếc bát trên bàn — khó xác định biên không có nghĩa là prediction sai. | Vẽ Polygon theo đường viền phần nhìn thấy, không gộp các cá thể và ghi chú vùng không rõ. | Kiểm tra độ chính xác của Mask, ranh giới giữa các vật thể và các trường hợp gán nhãn sai hoặc thiếu. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   Không chia sẻ, sao chép hoặc tải dữ liệu lab lên các nền tảng bên ngoài khi chưa được cấp phép; chỉ sử dụng dữ liệu đúng mục đích và trong phạm vi được giao.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   Tôi sẽ dừng xử lý, không tự ý chỉnh sửa hoặc chia sẻ dữ liệu và báo ngay cho Lab Coach hoặc người phụ trách dự án để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS` (cần xác nhận trực tiếp trên Colab)
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
