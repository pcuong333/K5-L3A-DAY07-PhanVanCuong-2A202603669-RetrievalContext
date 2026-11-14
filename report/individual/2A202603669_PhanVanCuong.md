# Báo cáo Day 7 — 2A202603669-Phan Van Cuong

## Dữ liệu và benchmark

- **Nguồn:** OpenStax, *Introduction to Computer Science*, OpenStax license.
- **Corpus:** `data/day6/introduction_to_computer_science.pdf`, trang vật lý 30–80, 51 trang có text.
- **Chunk:** strategy B, gom theo paragraph tối đa 800 ký tự; 206 chunks.
- **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, local, 384 chiều, chuẩn hóa vector.
- **Vector store:** PostgreSQL 16 + pgvector 0.8.7, Docker local, cosine distance.
- **Bảng:** `ingestion_chunks`, strategy B.
- **Kết nối:** `127.0.0.1:55433`, database `day6_lab`; Docker Compose và `DATABASE_URL` giữ ở Day 06, không commit `.env`.
- **Filter:** câu hỏi có metadata `section` được truyền vào JSONB filter của strategy B/C; các câu không có filter dùng toàn bộ strategy B.
- **Benchmark:** 15 câu có đáp án và 2 câu thiếu evidence ở dev; 5 câu có đáp án và 2 câu thiếu evidence ở test.
- **Dependencies:** `starter/requirements.txt`; PostgreSQL/pgvector chạy bằng Docker Compose của Day 06.

Các câu hỏi được chuyển từ Day 06. Một số evidence dài bị cắt qua ranh giới chunk; evidence trong bộ Day 7 được rút về đoạn liên tục nằm trong chunk để validator kiểm chứng được.

## Retrieval và thử nghiệm

| Cấu hình | Split | Hit@1 | Evidence Hit@3 | Context budget | Median ms |
|---|---|---:|---:|---:|---:|
| TF-IDF | dev | 5/15 | 7/15 | 800 ký tự | 0.13 |
| Dense Day 6, strategy B | dev | 0/15 | 2/15 | 800 ký tự | 46.07 |
| Hybrid RRF, k=60 | dev | 7/15 | 10/15 | 800 ký tự | 36.16 |

Hybrid kết hợp top-10 TF-IDF và top-10 dense bằng reciprocal rank fusion, sau đó lấy top-3.

## Agent và cải thiện

Agent dùng DeepSeek `deepseek-chat`, hybrid retrieval, tối đa 2 lượt search và top-k=3.

| Cấu hình | Grounded answer 2/15 | Citation hợp lệ | Abstention đúng | Từ chối nhầm |
|---|---:|---:|---:|---:|
| Budget 1200, dev | 11/15 | 11/11 | 2/2 | 4/15 |
| **Budget 2000, dev** | **12/15** | **17/17** | **2/2** | **3/15** |
| Budget 2000, test | 3/5 | 4/4 | 2/2 | 2/5 |

Chọn budget 2000 vì tăng số câu trả lời grounded từ 11 lên 12 và giảm false abstention từ 4 xuống 3 trên dev.

### Phân tích lỗi dev

- **q09:** Ở budget 1200 agent từ chối dù câu trả lời nằm trong đoạn về định nghĩa algorithm. Khi tăng budget lên 2000, context giữ được đoạn `openstax_cs-p56-paragraph-0` và agent trả lời đúng. Điều này ủng hộ giả thuyết context quá ngắn làm mất bằng chứng.
- **q11:** Agent vẫn từ chối ở budget 2000. Dense/hybrid đưa các đoạn trang 71, 77 và 62 vào candidate nhưng context cuối không làm nổi bật đoạn evidence ở trang 62 theo cách model cần. Cần thử query transformation hoặc filter-aware rerank; không dùng test để chỉnh sau khi đã chốt.

## Đánh giá test

- Agent test: 7/7 câu chạy thành công, không có lỗi provider.
- Tool use: 7/7 câu có gọi retrieval.
- Grounded answer: mức 2 là 3/5; mức 0 là 2/5 do từ chối nhầm.
- Citation hợp lệ: 4/4.
- Abstention câu thiếu evidence: 2/2.
- Baseline TF-IDF test: Hit@1 3/5, Evidence Hit@3 4/5.

## Trace và lệnh chạy

- Dev trace: `starter/results/deepseek-hybrid-dev-b2000/traces.json`
- Dev summary: `starter/results/deepseek-hybrid-dev-b2000/summary.json`
- Test trace: `starter/results/deepseek-hybrid-test-b2000/traces.json`
- Test summary: `starter/results/deepseek-hybrid-test-b2000/summary.json`
- Chi phí API: chưa đo token usage/cost theo từng request; các kết quả trên không nên được diễn giải là chi phí bằng 0.
- Khởi động vector store trước khi chạy: `docker compose up -d` trong `K5-L3A-DAY06-PhanVanCuong-2A202603669-DataIngestion\starter`.

```powershell
$day6="C:\Users\Admin\OneDrive\Desktop\pro\K5-L3A-DAY06-PhanVanCuong-2A202603669-DataIngestion"
$day7="C:\Users\Admin\OneDrive\Desktop\pro\K5-L3A-DAY07-PhanVanCuong-2A202603669-RetrievalContext"
$env:PYTHONPATH="$day7\starter"
& "$day6\starter\.venv\Scripts\python.exe" starter\baseline.py --chunks data\day6\converted\chunks.json --questions data\day6\questions_answerable.json --split dev --output starter\results\day6-tfidf-dev
& "$day6\starter\.venv\Scripts\python.exe" starter\baseline.py --chunks data\day6\converted\chunks.json --questions data\day6\questions_answerable.json --split test --output starter\results\day6-tfidf-test
& "$day6\starter\.venv\Scripts\python.exe" starter\run_deepseek_agent.py --split dev --budget 2000 --max-searches 2
& "$day6\starter\.venv\Scripts\python.exe" starter\run_deepseek_agent.py --split test --budget 2000 --max-searches 2
```

## Giới hạn

Corpus chỉ gồm trang 30–80 và benchmark test nhỏ. Context packing hiện ưu tiên candidate đứng trước nên có thể bỏ evidence ở chunk xếp hạng sau. Một số câu test bị agent từ chối dù có evidence; cần thử query expansion hoặc cho phép lượt search thứ hai để cải thiện.
