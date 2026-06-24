from collections import Counter


class PlateCache:
    """
    Per-track voting cache untuk stabilisasi hasil OCR plat nomor.

    Alur:
    - Setiap hasil OCR valid dimasukkan ke counter kandidat.
    - Pemenang = kandidat dengan vote terbanyak.
    - Setelah lock_threshold tercapai, hasil dikunci dan OCR dihentikan.
    """

    def __init__(
        self,
        lock_threshold: int = 5,
        min_votes_to_show: int = 2,
        max_candidates: int = 20,
    ):
        self.lock_threshold = lock_threshold
        self.min_votes_to_show = min_votes_to_show
        self.max_candidates = max_candidates

        # {track_id: {"counter": Counter, "locked": str|None, "crop": ndarray, "last_frame": int}}
        self._data: dict[int, dict] = {}

    # ------------------------------------------------------------------

    def _entry(self, tid: int) -> dict:
        if tid not in self._data:
            self._data[tid] = {
                "counter":    Counter(),
                "locked":     None,
                "crop":       None,
                "last_frame": -9999,
            }
        return self._data[tid]

    # ------------------------------------------------------------------

    def is_locked(self, tid: int) -> bool:
        return self._entry(tid)["locked"] is not None

    def last_frame(self, tid: int) -> int:
        return self._entry(tid)["last_frame"]

    def get_plate(self, tid: int) -> str | None:
        entry = self._entry(tid)
        if entry["locked"]:
            return entry["locked"]
        if not entry["counter"]:
            return None
        best, votes = entry["counter"].most_common(1)[0]
        return best if votes >= self.min_votes_to_show else None

    def get_crop(self, tid: int):
        return self._entry(tid)["crop"]

    # ------------------------------------------------------------------

    def update(
        self,
        tid: int,
        plate_text: str | None,
        crop,
        frame_idx: int,
    ) -> None:
        entry = self._entry(tid)
        entry["last_frame"] = frame_idx

        # Selalu update crop dengan yang terbaru (untuk thumbnail)
        if crop is not None and crop.size > 0:
            entry["crop"] = crop

        # Kalau sudah terkunci, abaikan hasil baru
        if entry["locked"]:
            return

        # Hanya masukkan hasil yang valid (tidak None / kosong)
        if not plate_text:
            return

        entry["counter"][plate_text] += 1

        # Cek apakah pemenang sudah mencapai lock_threshold
        best, votes = entry["counter"].most_common(1)[0]
        if votes >= self.lock_threshold:
            entry["locked"] = best

        # Batasi ukuran counter agar tidak membengkak
        if len(entry["counter"]) > self.max_candidates:
            # Buang kandidat dengan vote paling sedikit
            least = entry["counter"].most_common()[:-self.max_candidates - 1:-1]
            for candidate, _ in least:
                del entry["counter"][candidate]

    # ------------------------------------------------------------------

    def should_run_ocr(self, tid: int, frame_idx: int, interval: int) -> bool:
        """Return True jika OCR perlu dijalankan untuk track ini."""
        if self.is_locked(tid):
            return False
        return (frame_idx - self.last_frame(tid)) >= interval

    # ------------------------------------------------------------------

    def cleanup(self, active_ids: set[int], max_age: int = 60) -> None:
        """Hapus entry track yang sudah lama tidak aktif."""
        dead = [
            tid for tid in self._data
            if tid not in active_ids
        ]
        for tid in dead:
            del self._data[tid]