# Mạng Lưới Chú Ý Đồ Thị Có Đặc Trưng Cạnh (EGAT)

## Tác giả

- **Jun Chen** – Trường Phần Mềm, Đại học Giao Thông Thượng Hải, Trung Quốc (thunderboy@sjtu.edu.cn)
- **Haopeng Chen** – Trường Phần Mềm, Đại học Giao Thông Thượng Hải, Trung Quốc (chen-hp@sjtu.edu.cn)

## Tóm tắt

Trong những năm gần đây, nhiều mô hình mạng nơ-ron nhân tạo đã được đề xuất để giải quyết các bài toán học trên dữ liệu có cấu trúc đồ thị. Tuy nhiên, phần lớn các mô hình này chỉ tập trung vào đặc trưng của nút (node) mà bỏ qua hoặc đơn giản hóa đặc trưng của cạnh (edge), trong khi cạnh cũng đóng vai trò quan trọng không kém trong nhiều bài toán thực tế.

Trong bài báo này, chúng tôi đề xuất **Mạng lưới chú ý đồ thị có đặc trưng cạnh** (Edge-Featured Graph Attention Networks - **EGATs**), mở rộng từ **Mạng lưới chú ý đồ thị** (Graph Attention Networks - **GATs**) để xử lý các đồ thị có cả đặc trưng nút và cạnh. Mô hình EGATs cải tiến cấu trúc và cơ chế học của GATs, cho phép:

- Nhận cả đặc trưng của nút và cạnh làm đầu vào.
- Tích hợp thông tin cạnh vào quá trình tính toán trọng số chú ý.
- Cập nhật đồng thời đặc trưng của nút và cạnh một cách tương tác.

Thử nghiệm cho thấy EGATs có hiệu suất cạnh tranh với các mô hình phân loại nút tiên tiến hiện nay và đặc biệt hiệu quả đối với các bài toán có đặc trưng cạnh phong phú.

---

## 1. Giới thiệu

Trong nhiều ứng dụng thực tế, dữ liệu có thể được biểu diễn dưới dạng đồ thị, trong đó nút biểu diễn thực thể và cạnh biểu diễn mối quan hệ giữa các thực thể. Các phương pháp học trên đồ thị đã phát triển mạnh mẽ trong thời gian qua, với nhiều mạng nơ-ron đồ thị (**Graph Neural Networks - GNNs**) được đề xuất:

- **Mạng Nơ-ron Chập Đồ Thị (GCN)** (Kipf et al. [10]): Sử dụng lý thuyết phổ đồ thị để trích xuất đặc trưng nút.
- **Mạng Chú Ý Đồ Thị (GAT)** (Veličković et al. [21]): Ứng dụng cơ chế self-attention để tổng hợp đặc trưng nút một cách linh hoạt.

Dù đạt được nhiều thành tựu trong bài toán phân loại nút, các mô hình này vẫn có hạn chế lớn: **không tận dụng thông tin cạnh**. Tuy nhiên, trong nhiều bài toán thực tế, đặc trưng cạnh đóng vai trò quan trọng không kém đặc trưng nút. Ví dụ:

- **Phát hiện gian lận tài chính:** Các giao dịch (cạnh) chứa thông tin quan trọng hơn đặc trưng người dùng (nút).
- **Mạng xã hội:** Mối quan hệ giữa các người dùng có thể quan trọng hơn đặc trưng cá nhân của họ.

### Mô hình EGATs giải quyết vấn đề này như thế nào?

- Mở rộng cơ chế chú ý để **tích hợp đặc trưng cạnh** vào trọng số chú ý.
- **Cập nhật đặc trưng của cả nút và cạnh** đồng thời, thay vì chỉ tập trung vào nút như GATs.
- **Chiến lược hợp nhất đa cấp (multi-scale merging)** giúp tận dụng thông tin từ nhiều tầng khác nhau.

Đây là lần đầu tiên một mô hình **đối xử cạnh như thực thể ngang hàng với nút**, giúp cải thiện khả năng học của mạng nơ-ron đồ thị trong các bài toán thực tế.

---

## 4. Mô Hình Đề Xuất

### 4.1 Tổng quan về lớp EGAT

Một lớp EGAT (Edge-Featured Graph Attention Network) bao gồm hai khối chính:

- **Khối chú ý nút (Node Attention Block)**
- **Khối chú ý cạnh (Edge Attention Block)**

Lớp EGAT được thiết kế theo sơ đồ đối xứng, đảm bảo rằng cả đặc trưng của nút và cạnh đều được cập nhật song song và đồng bộ. Hình 1(a) minh họa cách một lớp EGAT hoạt động.

**Đầu vào của lớp EGAT:**

- **Tập hợp đặc trưng của nút**: $H = \{ \vec{h}_1, \vec{h}_2, \dots, \vec{h}_N \}$, với mỗi $\vec{h}_i \in \mathbb{R}^{F_H}$.
- **Tập hợp đặc trưng của cạnh**: $E = \{ \vec{e}_1, \vec{e}_2, \dots, \vec{e}_M \}$, với mỗi $\vec{e}_p \in \mathbb{R}^{F_E}$.
- $N$ và $M$ lần lượt là số lượng nút và cạnh trong đồ thị.
- $F_H$ và $F_E$ là số lượng đặc trưng của nút và cạnh tương ứng.

**Đầu ra của lớp EGAT:**

Sau khi xử lý, lớp EGAT tạo ra:

- **Bộ đặc trưng mới của nút**: $H' = \{ \vec{h}_1', \vec{h}_2', \dots, \vec{h}_N' \}$, với $\vec{h}_i' \in \mathbb{R}^{F_H'}$.
- **Bộ đặc trưng mới của cạnh**: $E' = \{ \vec{e}_1', \vec{e}_2', \dots, \vec{e}_M' \}$, với $\vec{e}_p' \in \mathbb{R}^{F_E'}$.

Do các phép biến đổi tuyến tính khác nhau áp dụng lên đặc trưng của nút và cạnh, số chiều của đặc trưng sau khi biến đổi $(F_H', F_E')$ có thể khác với số chiều ban đầu $(F_H, F_E)$.

(Phần tiếp theo sẽ mô tả về **Khối Chú Ý Nút (Node Attention Block)** và **Khối Chú Ý Cạnh (Edge Attention Block)** chi tiết hơn.)

## 4.2 Khối Chú Ý Nút (Node Attention Block)

Khối chú ý nút có nhiệm vụ tạo ra tập hợp đặc trưng nút mới $H'$, sử dụng cả thông tin từ nút và cạnh.

### Xử lý đặc trưng cạnh trước khi sử dụng

- Vì các cạnh trong tập $E$ có thứ tự sắp xếp cố định, nên rất khó xác định mối quan hệ giữa chúng với các nút liên quan.
- Để khắc phục, một phép biến đổi ánh xạ (mapping transformation) sẽ được áp dụng, tạo ra một tập đặc trưng cạnh mới $E^*$, trong đó mỗi phần tử $\vec{e}_{ij}$ đại diện cho cạnh nối giữa hai nút $i, j$.
- Quá trình này được thực hiện bằng ma trận ánh xạ cạnh $M_E$, là một tensor có kích thước $N \times N \times M$.

### Tính toán trọng số chú ý

Sau khi ánh xạ, trọng số chú ý $\alpha_{ij}$ giữa nút $i$ và nút $j$ được tính toán, đồng thời xét cả đặc trưng của cạnh kết nối chúng.

- Tập $N_i$ gồm các nút lân cận cấp 1 của $i$, bao gồm cả chính $i$.
- Các đặc trưng của nút $i$, nút $j$, và cạnh nối giữa chúng được nối (concatenate) và sử dụng để tính trọng số chú ý thông qua hàm kích hoạt LeakyReLU, sau đó chuẩn hóa bằng hàm softmax:

```math 
\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\vec{a}^T [\vec{h}_i \| \vec{h}_j \| \vec{e}_{ij}]\right)\right)}{\sum_{k \in N_i} \exp\left(\text{LeakyReLU}\left(\vec{a}^T [\vec{h}_i \| \vec{h}_k \| \vec{e}_{ik}]\right)\right)}
```

trong đó:
- $N_i$ là tập các nút lân cận cấp 1 của $i$ (bao gồm cả $i$).
- $\vec{a}$ là vector trọng số có kích thước $\mathbb{R}^{3F_H}$.

### Tạo đặc trưng nút mới

Sau khi có trọng số chú ý $\alpha_{ij}$, đặc trưng nút mới được tính bằng tổng có trọng số của các đặc trưng lân cận:

Đặc trưng nút mới $\vec{h}_i'$ được tính bằng tổng có trọng số:

```math
\vec{h}_i' = \sigma\left(\sum_{j \in N_i} \alpha_{ij} \vec{h}_j\right)
```

trong đó $\sigma$ là hàm phi tuyến (ví dụ: ReLU).

### Tạo đặc trưng nút tích hợp từ cạnh

Ngoài ra, khối chú ý nút còn tính toán một bộ đặc trưng nút tích hợp cạnh $H_m$, dùng trong lớp hợp nhất cuối cùng:


```math
\vec{m}_i = \sigma \left( \sum_{j \in N_i} \alpha_{ij} (\vec{h}_j || \vec{e}_{ij}) \right)
```

Tuy nhiên, $H_m$ chỉ được sử dụng ở tầng hợp nhất (merge layer) và không được truyền vào lớp EGAT tiếp theo.

---

## 4.3 Khối Chú Ý Cạnh (Edge Attention Block)
* Graph transformation:
        * The nodes and edges’ roles are inversed 
        * We consider two edges are adjacent only if two edges have at least one common vertex
        ![](https://i.imgur.com/dIXAqYX.png)
        
Khối chú ý cạnh được thiết kế để cập nhật đặc trưng của các cạnh, đóng vai trò quan trọng tương đương với nút trong quá trình tính toán trọng số chú ý.

### Tính trọng số chú ý cho cạnh

Trọng số chú ý giữa cạnh $p$ và các cạnh lân cận $q$ được tính toán như sau:

```math
\beta_{pq} = \frac{\exp(\text{LeakyReLU}(\vec{b}^T [\vec{e}_p || \vec{e}_q || \vec{h}_{pq}]))}{\sum_{k \in N_p} \exp(\text{LeakyReLU}(\vec{b}^T [\vec{e}_p || \vec{e}_k || \vec{h}_{pk}]))}
```

- $N_p$: Tập các cạnh lân cận cấp 1 của cạnh $p$ (bao gồm chính $p$).
- $\vec{b}$: Vector trọng số có kích thước $\mathbb{R}^{2F_E' + F_H'}$.

### Tính đặc trưng cạnh mới

Đặc trưng cạnh mới $\vec{e}_p'$ được tính thông qua tổng có trọng số của các đặc trưng cạnh lân cận:

```math
\vec{e}_p' = \sigma \left( \sum_{q \in N_p} \beta_{pq} \vec{e}_q \right)
```

Như đặc trưng nút, đặc trưng cạnh mới sẽ phản ánh mối quan hệ giữa các cạnh, cũng như thông tin được tích hợp từ các nút liên quan.

---

## 4.4 Kiến Trúc EGAT (EGAT Architecture)
* EGAT Architecture
    * Stacking several EGAT layers and appending a merge layer at the tail
        ![](https://i.imgur.com/rJVaPLt.png)
Mô hình EGATs được xây dựng bằng cách xếp chồng nhiều lớp EGAT và thêm một lớp hợp nhất (merge layer) ở cuối. Kiến trúc này được minh họa trong Hình 1(b).

### Chiến lược hợp nhất đa tầng (Multi-scale Merge Strategy)

- Chiến lược hợp nhất đa tầng được sử dụng để tổng hợp các đặc trưng từ nhiều lớp khác nhau, tương tự như trong các mô hình CNN.
- EGATs không chỉ thu thập đặc trưng của nút mà còn cả đặc trưng của cạnh.
- Tất cả các $H_m$ được tạo ra từ các lần lặp khác nhau sẽ được tổng hợp lại bằng cách nối (concatenate).

### Cơ chế chú ý đa đầu (Multi-head Attention)

- Khác với GATs, cơ chế chú ý đa đầu trong EGATs được thực hiện trên toàn bộ các lớp EGAT thay vì chỉ một lớp duy nhất.
- $K$ bộ đặc trưng tích hợp cạnh độc lập sẽ được tính toán và hợp nhất, tạo thành biểu diễn đặc trưng tổng thể:

```math
\vec{h}_i^* = \bigg\|_{k=1}^{K} \bigg\|_{l=1}^{L} m_{il,k}
```

trong đó:

- $L$ là số lượng lớp EGAT.
- $m_{il,k}$ đại diện cho đặc trưng nút tích hợp cạnh của nút $i$ được tạo ra trong lần lặp $l$ của nhóm $k$.

### Tổng kết kiến trúc EGATs

- Sau khi hợp nhất các đặc trưng tích hợp cạnh, mô hình sử dụng một phép chập một chiều (1D convolution) để chuyển đổi tuyến tính và phi tuyến các đặc trưng.
- Trong các bài toán phân loại nút, một hàm softmax sẽ được áp dụng cuối cùng để tạo ra nhãn dự đoán.

**Kết luận:**

- Khối chú ý cạnh (Edge Attention Block) trong EGATs đảm bảo rằng đặc trưng của cạnh cũng được cập nhật và học hỏi từ các cạnh lân cận.
- Chiến lược hợp nhất đa tầng giúp EGATs tổng hợp thông tin từ nhiều tầng khác nhau, tạo ra biểu diễn đặc trưng phong phú cho cả nút và cạnh.
- Cơ chế chú ý đa đầu giúp mô hình ổn định hơn trong việc học từ dữ liệu đồ thị có đặc trưng phức tạp.

Những cải tiến này làm cho EGATs trở thành một mô hình mạnh mẽ, có thể áp dụng hiệu quả trong các bài toán học trên đồ thị với thông tin từ cả nút và cạnh.

## 5. Thí Nghiệm

### 5.1 Bộ Dữ Liệu
- **Đồ thị nhạy cảm nút (Node-sensitive graphs)**:
  - Đặc trưng nút tương quan mạnh với nhãn nút
  - Bao gồm: Cora, Citeseer, Pubmed
  - Đồ thị vô hướng và **không có đặc trưng cạnh**
- **Đồ thị nhạy cảm cạnh (Edge-sensitive graphs)**:
  - Dữ liệu tài chính hợp tác từ hồ sơ giao dịch thực tế:
    - **Trade-B**: 3907 nút (97 nút có nhãn), 4394 cạnh - bài toán phân loại nhị phân
    - **Trade-M**: 4431 nút (139 nút có nhãn), 4900 cạnh - bài toán phân loại ba lớp
  - Mỗi nút biểu diễn khách hàng với thuộc tính mức độ rủi ro
  - Mỗi cạnh biểu diễn quan hệ giao dịch, chứa đặc trưng về số lượng và tổng giá trị giao dịch
  - Ban đầu là đồ thị có hướng, được chuyển thành vô hướng để phù hợp với EGATs

### 5.2 Thiết Lập Thí Nghiệm
- Tham số mô hình: $L=2$ (số lớp), $K=8$ (số đầu chú ý)
- **Đồ thị nhạy cảm nút**:
  - Tạo đặc trưng cạnh yếu bằng cách đếm số cạnh kề liền kề
  - Kích thước đặc trưng: $F'_H=8$ (nút), $F'_E=4$ (cạnh)
- **Đồ thị nhạy cảm cạnh**:
  - Thử nghiệm với tỷ lệ đặc trưng nút-cạnh khác nhau: $8:4$, $6:6$, $4:8$

### 5.3 Kết Quả
#### a) Đồ thị nhạy cảm nút
![](https://i.imgur.com/YzZBtC5.png)
- **Nhận xét**:
  - Độ chính xác giảm nhẹ (~0.5-1%) so với GAT trên Cora và Citeseer do nhiễu từ đặc trưng cạnh nhân tạo
  - Tuy nhiên, ảnh hưởng tiêu cực không đáng kể, chứng tỏ EGATs có khả năng lọc nhiễu tốt

#### b) Đồ thị nhạy cảm cạnh
![](https://i.imgur.com/FZraaA0.png)
- **Thiết lập so sánh**:
  - GAT cơ bản: Chỉ sử dụng đặc trưng nút
  - Biến thể GAT: Tích hợp đặc trưng cạnh vào nút bằng phép **sum**, **average**, và **max pooling**
- **Kết luận**:
  - EGATs đạt độ chính xác vượt trội (~5-8%) so với mọi biến thể GAT
  - Khi đặc trưng cạnh quan trọng hơn nút, tỷ lệ $F'_H$ : $F'_E$ nhỏ (ví dụ $4:8$) cho kết quả tốt nhất
  - EGATs là phương pháp đầu tiên xử lý hiệu quả đồ thị có đặc trưng cạnh phức tạp
