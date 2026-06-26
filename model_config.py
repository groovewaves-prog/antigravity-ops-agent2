# -*- coding: utf-8 -*-
"""
モデル設定（Gemini API / generativelanguage v1beta）
=====================================================
散在していたモデルIDをここに集約する。モデルを変えるときは、
このファイルの定数だけを書き換えればよい（各モジュールはここを参照する）。

利用可能なモデルIDは Google AI Studio、もしくは ListModels で確認できる
（このファイル末尾の list_available_models() を実行）。

Gemma 4（2026-04 提供開始）のホスト版ID：
  - gemma-4-26b-a4b-it   … 26B（A4B / MoE）。コスト・速度と性能のバランス型（既定）
  - gemma-4-31b-it       … 31B。最も高性能（RCAの推論を重視するならこちら）
  - gemma-4-e4b-it       … 軽量版（低リソース向け）

旧 ID（gemma-3-12b-it / gemini-1.5-flash）はホストAPIで提供終了に向かっており、
generateContent で 404 になるため使用しない。
"""

# チャット／ネットワーク分析などに使う Gemma モデル
GEMMA_MODEL = "gemma-4-26b-a4b-it"

# RCA（推論）フォールバックに使うモデル。
# 既定では Gemma に一本化している（モデル系統を1つに揃え、保守を減らすため）。
# Gemini を使いたい場合は、現行IDへ変更する（例: "gemini-3.5-flash"）。
REASONING_MODEL = GEMMA_MODEL


def list_available_models(api_key: str):
    """generateContent をサポートする利用可能なモデルID一覧を返す。
    「404 model not found」の切り分けに使う（エラーが促す ListModels 相当）。"""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    names = []
    for m in genai.list_models():
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            names.append(m.name)
    return names


if __name__ == "__main__":
    import os
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        print("GOOGLE_API_KEY が未設定です。環境変数を設定して再実行してください。")
    else:
        print("generateContent 対応モデル:")
        for n in list_available_models(key):
            print("  ", n)
