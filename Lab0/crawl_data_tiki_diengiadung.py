import requests
import pandas as pd
import time
import sys

# Cai dat cho phep doc tieng Viet 
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Tao User Agent
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}

# Ham crawl du lieu tu Tiki danh muc Dien Gia Dung (category=1882)
def crawl_tiki_diengiadung(so_trang=6, limit=40):
    all_products = []
    
    # Vong lap dem so trang 
    for page in range(1, so_trang + 1):
        url = f"https://tiki.vn/api/personalish/v1/blocks/listings?category=1882&page={page}&limit={limit}"
        res = requests.get(url, headers=HEADERS, timeout=10)
        items = res.json().get("data", [])
        
        for it in items:
            sold_info = it.get("quantity_sold") or {}
            sold = sold_info.get("value", 0) if isinstance(sold_info, dict) else 0
            
            # Them cac thong tin cua san pham dien gia dung
            all_products.append({
                "id": len(all_products) + 1,
                "ten_san_pham": it.get("name"),
                "thuong_hieu": it.get("brand_name") or "Không rõ",
                "gia_ban": it.get("price", 0),
                "gia_goc": it.get("original_price", 0),
                "pham_tram_giam_gia": it.get("discount_rate", 0),
                "chi_so_danh_gia": it.get("rating_average", 0.0),
                "so_luong_binh_luan": it.get("review_count", 0),
                "so_luong_ban": sold,
            })
        time.sleep(1)

    # Chuyen du lieu sang DataFrame
    df = pd.DataFrame(all_products)
    
    # Phan loai du lieu dua theo danh gia, so luong ban va gia (danh cho dien gia dung)
    df["nhan_danh_gia"] = df["chi_so_danh_gia"].apply(lambda r: "xuat_sac" if r >= 4.8 else ("tot" if r >= 4.5 else "trung_binh"))
    df["nhan_trang_thai"] = df["so_luong_ban"].apply(lambda s: "ban_chay" if s >= 500 else ("tiem_nang" if s >= 50 else "moi_ban"))
    df["nhan_gia_ca"] = df["gia_ban"].apply(lambda p: "gia_re" if p < 500000 else ("tam_trung" if p <= 2000000 else "cao_cap"))

    # Luu du lieu ra file csv   
    df.to_csv("tiki_diengiadung.csv", index=False, encoding="utf-8-sig")
    return df

if __name__ == "__main__":
    df = crawl_tiki_diengiadung(so_trang=6) 
    print("Thu thap thanh cong {} san pham.".format(len(df)))
