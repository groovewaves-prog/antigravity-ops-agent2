# -*- coding: utf-8 -*-
"""
AIOps Agent - Global Rate Limiter Module (v3 - 遅延解消版)
==========================================================
gemma-3-12b-it の制限:
- 30 RPM / 14,400 RPD
"""

import time
import threading
import logging
from typing import Optional, Dict, Any
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """レート制限の設定"""
    rpm: int = 30
    rpd: int = 14400
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
    
    def _check_limits(self) -> tuple[bool, float]:
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
