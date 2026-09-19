from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(storage={'root_dir': 'scraped_thai_receipts'})

keywords = [
    "ใบเสร็จ 7-11",
    "ใบเสร็จรับเงิน ค่าไฟ",
    "ใบเสร็จ Lotus",
    "ใบเสร็จ Big C",
    "ใบกำกับภาษีอย่างย่อ",
    "ใบเสร็จ ค่าน้ำประปา",
    "thai receipt"
]

for kw in keywords:
    crawler.crawl(keyword=kw, max_num=30, file_idx_offset='auto')