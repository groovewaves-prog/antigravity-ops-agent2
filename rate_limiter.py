# -*- coding: utf-8 -*-
"""
AIOps Agent - Global Rate Limiter Module
=========================================
全モジュール共通のレートリミッター

gemma-3-12b-it の制限:
- 30 RPM (1分あたり30リクエスト)
- 14,400 RPD (1日あたり14,400リクエスト)
- 128,000 トークン (入力コンテキスト)
- 8,192 トークン (出力)
"""

import time
import threading
import logging
from typing import Optional, Dict, Any, Callable
from collections import deque
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import json

logger = logging.getLogger(__name__)

# =====================================================
# 設定定数
# =====================================================
@dataclass
class RateLimitConfig:
    """レート制限の設定"""
    rpm: int = 30                    # Requests Per Minute
    rpd: int = 14400                 # Requests Per Day
    input_tokens: int = 128000       # 入力トークン上限
    output_tokens: int = 8192        # 出力トークン上限
    safety_margin: float = 0.8       # 安全マージン (80%で制限)
    retry_base_delay: float = 2.0    # リトライ基本待機時間
    retry_max_delay: float = 60.0    # リトライ最大待機時間
    cache_ttl: int = 3600            # キャッシュTTL (秒)


# =====================================================
# グローバルレートリミッター (シングルトン)
# =====================================================
class GlobalRateLimiter:
    """
    スレッドセーフなグローバルレートリミッター
    
    特徴:
    - トークンバケットアルゴリズム
    - 429/503エラー時の指数バックオフ
    - レスポンスキャッシュ
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
        if self._initialized:
            return
        
        self.config = config or RateLimitConfig()
        self._request_times: deque = deque(maxlen=self.config.rpm)
        self._daily_count: int = 0
        self._daily_reset_time: float = time.time()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._initialized = True
        
        logger.info(f"GlobalRateLimiter initialized: {self.config.rpm} RPM, {self.config.rpd} RPD")
    
    def _clean_expired_cache(self):
        """期限切れキャッシュを削除"""
        now = time.time()
        with self._cache_lock:
            expired_keys = [
                k for k, v in self._cache.items() 
                if now - v.get('timestamp', 0) > self.config.cache_ttl
            ]
            for k in expired_keys:
                del self._cache[k]
    
    def get_cache(self, key: str) -> Optional[Any]:
        """キャッシュから取得"""
        self._clean_expired_cache()
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry and time.time() - entry['timestamp'] < self.config.cache_ttl:
                logger.debug(f"Cache hit: {key[:16]}...")
                return entry['value']
        return None
    
    def set_cache(self, key: str, value: Any):
        """キャッシュに保存"""
        with self._cache_lock:
            self._cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
            logger.debug(f"Cache set: {key[:16]}...")
    
    def compute_cache_key(self, *args, **kwargs) -> str:
        """キャッシュキーを計算"""
        content = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_daily_limit(self) -> bool:
        """日次制限をチェック"""
        now = time.time()
        # 日付が変わったらリセット
        if now - self._daily_reset_time > 86400:
            self._daily_count = 0
            self._daily_reset_time = now
        
        effective_limit = int(self.config.rpd * self.config.safety_margin)
        return self._daily_count < effective_limit
    
    def _check_minute_limit(self) -> bool:
        """分間制限をチェック"""
        now = time.time()
        
        # 1分以上前のリクエストを削除
        while self._request_times and now - self._request_times[0] > 60:
            self._request_times.popleft()
        
        effective_limit = int(self.config.rpm * self.config.safety_margin)
        return len(self._request_times) < effective_limit
    
    def wait_for_slot(self, timeout: float = 120.0) -> bool:
        """
        リクエスト可能になるまで待機
        
        Returns:
            bool: リクエスト可能ならTrue
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self._request_lock:
                if self._check_daily_limit() and self._check_minute_limit():
                    return True
            
            # 待機時間を計算
            if self._request_times:
                oldest = self._request_times[0]
                wait_time = max(0, 60 - (time.time() - oldest)) + 0.1
            else:
                wait_time = 2.0
            
            wait_time = min(wait_time, timeout - (time.time() - start_time))
            if wait_time > 0:
                logger.info(f"Rate limit: waiting {wait_time:.1f}s for slot...")
                time.sleep(wait_time)
        
        logger.warning("Rate limit: timeout waiting for slot")
        return False
    
    def record_request(self):
        """リクエストを記録"""
        with self._request_lock:
            self._request_times.append(time.time())
            self._daily_count += 1
            logger.debug(f"Request recorded: {len(self._request_times)}/{self.config.rpm} RPM, {self._daily_count}/{self.config.rpd} RPD")
    
    def get_stats(self) -> Dict[str, Any]:
        """現在の統計を取得"""
        now = time.time()
        with self._request_lock:
            # 1分以内のリクエスト数
            recent_requests = sum(1 for t in self._request_times if now - t <= 60)
            
            return {
                'requests_last_minute': recent_requests,
                'requests_today': self._daily_count,
                'rpm_limit': self.config.rpm,
                'rpd_limit': self.config.rpd,
                'rpm_available': max(0, int(self.config.rpm * self.config.safety_margin) - recent_requests),
                'cache_size': len(self._cache)
            }


# =====================================================
# レートリミッター付きデコレータ
# =====================================================
def rate_limited(use_cache: bool = True, cache_key_func: Optional[Callable] = None):
    """
    LLM呼び出し関数用デコレータ
    
    Args:
        use_cache: キャッシュを使用するか
        cache_key_func: カスタムキャッシュキー生成関数
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = GlobalRateLimiter()
            
            # キャッシュチェック
            if use_cache:
                if cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                else:
                    cache_key = limiter.compute_cache_key(func.__name__, *args, **kwargs)
                
                cached = limiter.get_cache(cache_key)
                if cached is not None:
                    return cached
            
            # レート制限待機
            if not limiter.wait_for_slot():
                raise RuntimeError("Rate limit exceeded: timeout waiting for available slot")
            
            # リクエスト実行
            limiter.record_request()
            result = func(*args, **kwargs)
            
            # キャッシュ保存
            if use_cache and result is not None:
                limiter.set_cache(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


def rate_limited_with_retry(
    max_retries: int = 3,
    use_cache: bool = True,
    retry_on: tuple = (429, 503)
):
    """
    リトライ付きレートリミッターデコレータ
    
    Args:
        max_retries: 最大リトライ回数
        use_cache: キャッシュを使用するか
        retry_on: リトライ対象のHTTPステータスコード
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = GlobalRateLimiter()
            config = limiter.config
            
            # キャッシュチェック
            if use_cache:
                cache_key = limiter.compute_cache_key(func.__name__, *args, **kwargs)
                cached = limiter.get_cache(cache_key)
                if cached is not None:
                    return cached
            
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    # レート制限待機
                    if not limiter.wait_for_slot():
                        raise RuntimeError("Rate limit exceeded")
                    
                    # リクエスト実行
                    limiter.record_request()
                    result = func(*args, **kwargs)
                    
                    # キャッシュ保存
                    if use_cache and result is not None:
                        limiter.set_cache(cache_key, result)
                    
                    return result
                
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()
                    
                    # リトライ対象かチェック
                    should_retry = any(str(code) in error_msg for code in retry_on)
                    should_retry = should_retry or 'overloaded' in error_msg
                    should_retry = should_retry or 'rate' in error_msg
                    
                    if should_retry and attempt < max_retries:
                        delay = min(
                            config.retry_base_delay * (2 ** attempt),
                            config.retry_max_delay
                        )
                        logger.warning(f"Retry {attempt + 1}/{max_retries}: waiting {delay:.1f}s after error: {e}")
                        time.sleep(delay)
                    else:
                        break
            
            raise last_error
        
        return wrapper
    return decorator


# =====================================================
# ユーティリティ関数
# =====================================================
def get_rate_limiter() -> GlobalRateLimiter:
    """グローバルレートリミッターインスタンスを取得"""
    return GlobalRateLimiter()


def estimate_tokens(text: str) -> int:
    """
    トークン数を概算（日本語対応）
    
    簡易計算: 英語は4文字≒1トークン、日本語は1.5文字≒1トークン
    """
    if not text:
        return 0
    
    # 簡易的な判定
    ascii_count = sum(1 for c in text if ord(c) < 128)
    non_ascii_count = len(text) - ascii_count
    
    return int(ascii_count / 4 + non_ascii_count / 1.5)


def check_input_limit(prompt: str, safety_margin: float = 0.9) -> bool:
    """
    入力プロンプトがトークン制限内か確認
    
    Args:
        prompt: 入力プロンプト
        safety_margin: 安全マージン（デフォルト90%）
    
    Returns:
        bool: 制限内ならTrue
    """
    config = RateLimitConfig()
    estimated_tokens = estimate_tokens(prompt)
    limit = int(config.input_tokens * safety_margin)
    
    if estimated_tokens > limit:
        logger.warning(f"Input may exceed token limit: {estimated_tokens} > {limit}")
        return False
    return True
