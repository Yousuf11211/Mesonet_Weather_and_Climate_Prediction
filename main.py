from herbie import Herbie
from datetime import datetime, timedelta
from pathlib import Path
from multiprocessing import Pool, cpu_count
import os

start_date = datetime(2020, 1, 1, 0)
end_date = datetime(2025, 7, 7, 0)
forecast_hours = [0]
save_root = Path("D:/Herbie/HRRR_Large")
report_root = save_root.parent / "Reports_HRRR_Month_Year"
num_processes = min(cpu_count(), 4)

def get_report_path(date):
    return report_root / f"{date.year}" / f"{date.strftime('%B_%Y')}_report.txt"

def ensure_directories(path):
    path.mkdir(parents=True, exist_ok=True)

def download_day(date_fxx_tuple):
    date, fxx = date_fxx_tuple
    try:
        H = Herbie(
            date.strftime("%Y-%m-%d %H:%M"),
            model="hrrr",
            product="sfc",
            fxx=fxx,
            save_dir=save_root
        )


        filepath = H.get_localFilePath()
        downloaded = False
        size_mb = 0

        if not filepath.exists():
            H.download()
            downloaded = True

        if filepath.exists():
            old_folder = filepath.parent
            new_folder = save_root / "hrrr" / f"{date.year}" / date.strftime("%B") / date.strftime("%Y-%m-%d")

            if old_folder != new_folder:
                new_folder.parent.mkdir(parents=True, exist_ok=True)
                old_folder.rename(new_folder)

            filepath = new_folder / filepath.name
            size_mb = filepath.stat().st_size / 1e6
            status = "Downloaded" if downloaded else "✔️ Already exists"
        else:
            status = "File not found"

    except Exception as e:
        return date.strftime('%Y-%m-%d'), 0, f"Error: {str(e)}"

    print(f" {status}: {filepath.name} ({size_mb:.1f} MB)")
    return date.strftime('%Y-%m-%d'), size_mb, status

def process_month(year, month):
    total_size = 0
    month_start = datetime(year, month, 1)
    next_month = month_start.replace(day=28) + timedelta(days=4)
    month_end = next_month.replace(day=1) - timedelta(days=1)
    days = [month_start + timedelta(days=i) for i in range((month_end - month_start).days + 1)]
    tasks = [(d, fxx) for d in days for fxx in forecast_hours]

    with Pool(processes=num_processes) as pool:
        results = pool.map(download_day, tasks)

    report_lines = []
    missing_days = set()

    for day, size, status in results:
        report_lines.append(f"{day} - {status} - {size:.1f} MB")
        total_size += size
        if "Error" in status or "not found" in status:
            missing_days.add(day)

    report_text = "\n".join(report_lines)
    report_text += f"\n\nTotal for {month_start.strftime('%B %Y')}: {total_size:.1f} MB"

    if missing_days:
        report_text += "\n\nMissing days (failed or zero size downloads):"
        for md in sorted(missing_days):
            report_text += f"\n - {md}"

    report_path = get_report_path(month_start)
    ensure_directories(report_path.parent)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nCompleted {month_start.strftime('%B %Y')} - Total: {total_size:.1f} MB")
    if missing_days:
        print(f"Missing days: {', '.join(sorted(missing_days))}")
    print(f"Report saved to: {report_path}\n")

if __name__ == "__main__":
    current = start_date
    while current <= end_date:
        process_month(current.year, current.month)
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)

    print("\nAll downloads complete!")
