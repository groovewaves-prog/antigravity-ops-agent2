# -*- coding: utf-8 -*-
"""
AIOps Agent - Global Rate Limiter Module (v3 - Complete)
=========================================================
gemma-3-12b-it の制限:
- 30 RPM / 14,400 RPD
- 128,000 入力トークン / 8,192 出力トークン
"""

import time
import threading
import logging
from typing import Optional, Dict, Any, Callable
from collections import deque
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """レート制限の設定"""
    rpm: int = 30
    rpd: int = 14400
    input_tokens: int = 128000
    output_tokens: int = 8192
    safety_margin: float = 0.9
    cache_ttl: int = 3600


class GlobalRateLimiter:
    """
    スレッドセーフなグローバルレートリミッター（遅延解消版）
    
    ★改善ポイント:
    - 即時チェック: 制限内なら待機なしで即座にTrue返却
    - 最小待機: 制限超過時のみ必要最小限の待機
    """
    
    _instance: Optional['GlobalRateLimiter'] = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[RateLimitConfig] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        if getattr(self, '_initialized', False):
            return
        
        self.config = config or RateLimitConfig()
        self._request_times: deque = deque(maxlen=self.config.rpm * 2)
        self._daily_count: int = 0
        self._daily_reset_time: float = time.time()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._initialized = True
    
    def _clean_old_requests(self):
        """1分以上前のリクエストを削除"""
        now = time.time()
        while self._request_times and now - self._request_times[0] > 60:
            self._request_times.popleft()
    
    def _check_limits(self) -> tuple:
        """
        制限チェック
        
        Returns:
            (can_proceed, wait_time): 実行可否と必要な待機時間
        """
        now = time.time()
        
        # 日次リセット
        if now - self._daily_reset_time > 86400:
            self._daily_count = 0
            self._daily_reset_time = now
        
        # 古いリクエストを削除
        self._clean_old_requests()
        
        # 制限計算
        rpm_limit = int(self.config.rpm * self.config.safety_margin)
        rpd_limit = int(self.config.rpd * self.config.safety_margin)
        
        current_rpm = len(self._request_times)
        
        # 日次制限チェック
        if self._daily_count >= rpd_limit:
            return False, 3600  # 1時間待機（実質ブロック）
        
        # 分間制限チェック
        if current_rpm < rpm_limit:
            return True, 0  # ★即座に実行可能
        
        # 待機時間計算：最古のリクエストが1分経過するまで
        if self._request_times:
            oldest = self._request_times[0]
            wait_time = max(0.1, 60 - (now - oldest) + 0.1)
            return False, wait_time
        
        return True, 0
    
    def wait_for_slot(self, timeout: float = 60.0) -> bool:
        """
        リクエスト可能になるまで待機（遅延最小化版）
        
        ★改善: 制限内なら待機なしで即座にTrue返却
        """
        start_time = time.time()
        
        while True:
            with self._request_lock:
                can_proceed, wait_time = self._check_limits()
                
                if can_proceed:
                    return True  # ★即座に返却
            
            # タイムアウトチェック
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                return False
            
            # 必要最小限の待機
            actual_wait = min(wait_time, timeout - elapsed, 5.0)
            if actual_wait > 0:
                time.sleep(actual_wait)
    
    def record_request(self):
        """リクエストを記録"""
        with self._request_lock:
            self._request_times.append(time.time())
            self._daily_count += 1
    
    def get_cache(self, key: str) -> Optional[Any]:
        """キャッシュ取得"""
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry and time.time() - entry['ts'] < self.config.cache_ttl:
                return entry['val']
        return None
    
    def set_cache(self, key: str, value: Any):
        """キャッシュ設定"""
        with self._cache_lock:
            self._cache[key] = {'val': value, 'ts': time.time()}
    
    def get_stats(self) -> Dict[str, Any]:
        """統計取得"""
        with self._request_lock:
            self._clean_old_requests()
            return {
                'requests_last_minute': len(self._request_times),
                'rpm_limit': self.config.rpm,
                'daily_count': self._daily_count,
                'cache_size': len(self._cache)
            }


# =====================================================
# ユーティリティ関数（inference_engine.py用）
# =====================================================
def estimate_tokens(text: str) -> int:
    """
    テキストのトークン数を概算
    
    日本語は1文字≒1.5トークン、英語は1単語≒1.3トークン
    """
    if not text:
        return 0
    
    # 日本語文字数をカウント
    japanese_chars = sum(1 for c in text if '\u3000' <= c <= '\u9fff' or '\uff00' <= c <= '\uffef')
    # 英語単語数を概算
    english_words = len(text.split()) - japanese_chars // 2
    
    # 概算トークン数
    return int(japanese_chars * 1.5 + max(0, english_words) * 1.3)


def check_input_limit(text: str, limit: int = 100000) -> bool:
    """
    入力テキストがトークン制限内かチェック
    
    Args:
        text: チェック対象テキスト
        limit: トークン上限（デフォルト: 100,000 = 128,000の約80%）
    
    Returns:
        bool: 制限内ならTrue
    """
    return estimate_tokens(text) < limit


def rate_limited_with_retry(max_retries: int = 3, base_delay: float = 2.0):
    """
    レート制限とリトライを適用するデコレータ
    
    Args:
        max_retries: 最大リトライ回数
        base_delay: 基本待機時間（秒）
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = GlobalRateLimiter()
            
            for attempt in range(max_retries + 1):
                try:
                    if not limiter.wait_for_slot(timeout=30):
                        if attempt < max_retries:
                            time.sleep(base_delay * (attempt + 1))
                            continue
                        raise RuntimeError("Rate limit timeout")
                    
                    limiter.record_request()
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(x in error_msg for x in ['429', '503', 'overloaded', 'resource_exhausted']):
                        if attempt < max_retries:
                            wait_time = base_delay * (attempt + 1)
                            logger.warning(f"Rate limit error, retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                    raise
            
            return None
        return wrapper
    return decorator
