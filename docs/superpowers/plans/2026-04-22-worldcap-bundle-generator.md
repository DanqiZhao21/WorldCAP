# WorldCAP Bundle Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean, reproducible controller-bundle generator for `newCtrl` that can regenerate 64/128/1024 bundles with explicit source refs and validation output.

**Architecture:** Keep the historical `ControllerExp/scripts/genTest.py` intact for backward compatibility, and add a new generator module/script that factors out bundle generation logic into testable helpers. The new path will accept explicit input/output arguments, preserve source ref semantics by configuration, and emit validation statistics after writing the bundle.

**Tech Stack:** Python, NumPy, argparse, pytest, existing `PDMSimulator`/nuPlan/navsim components.

---
