from __future__ import annotations

import json
import os
import threading
import traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import joblib
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from needle_hook_features import FEATURE_NAMES, build_dataset, extract_features_from_csv, iter_csv_files, read_signal


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("针钩失效判断 GUI")
        self.geometry("1380x860")
        self.minsize(1180, 760)

        self.chart_files: dict[str, Path] = {}
        self.busy = False

        self._init_vars()
        self._build_ui()
        self.log("界面已就绪。")

    def _init_vars(self) -> None:
        self.train_valid = tk.StringVar(value="valid")
        self.train_invalid = tk.StringVar(value="invalid")
        self.train_model = tk.StringVar(value="model/needle_hook_model.joblib")
        self.train_metrics = tk.StringVar(value="model/metrics.json")
        self.train_chart_dir = tk.StringVar(value="model/training_figures")
        self.test_size = tk.StringVar(value="0.2")
        self.random_state = tk.StringVar(value="42")
        self.n_estimators = tk.StringVar(value="500")
        self.n_jobs = tk.StringVar(value="1")
        self.use_oob = tk.BooleanVar(value=True)
        self.oob_step = tk.StringVar(value="20")

        self.pred_model = tk.StringVar(value="model/needle_hook_model.joblib")
        self.pred_input = tk.StringVar(value="")
        self.pred_csv = tk.StringVar(value="model/predictions.csv")
        self.pred_chart_dir = tk.StringVar(value="model/prediction_figures")
        self.chart_key = tk.StringVar(value="")
        self.status = tk.StringVar(value="就绪")

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        nb = ttk.Notebook(left)
        nb.grid(row=0, column=0, sticky="nsew")
        train_tab = ttk.Frame(nb, padding=10)
        pred_tab = ttk.Frame(nb, padding=10)
        nb.add(train_tab, text="训练与图表")
        nb.add(pred_tab, text="预测与图表")

        self._build_train_tab(train_tab)
        self._build_pred_tab(pred_tab)

        bar = ttk.Frame(right)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        bar.grid_columnconfigure(1, weight=1)
        ttk.Label(bar, text="图表:").grid(row=0, column=0, sticky="w")
        self.combo = ttk.Combobox(bar, textvariable=self.chart_key, state="readonly")
        self.combo.grid(row=0, column=1, sticky="ew", padx=8)
        self.combo.bind("<<ComboboxSelected>>", lambda _: self.show_chart())
        ttk.Button(bar, text="显示", command=self.show_chart).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(bar, text="打开目录", command=self.open_chart_dir).grid(row=0, column=3)

        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.ax.set_title("训练/预测后可在这里预览图表")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        log_fr = ttk.Frame(self, padding=(10, 0, 10, 10))
        log_fr.grid(row=1, column=0, columnspan=2, sticky="nsew")
        log_fr.grid_columnconfigure(0, weight=1)
        ttk.Label(log_fr, text="日志").grid(row=0, column=0, sticky="w")
        self.log_box = tk.Text(log_fr, height=10, wrap="word")
        self.log_box.grid(row=1, column=0, sticky="ew")
        ttk.Label(log_fr, textvariable=self.status).grid(row=2, column=0, sticky="w")

    def _add_path_row(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, mode: str = "dir") -> None:
        parent.grid_columnconfigure(1, weight=1)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4, padx=(8, 8))

        def browse() -> None:
            if mode == "dir":
                p = filedialog.askdirectory(initialdir=".")
            elif mode == "file":
                p = filedialog.askopenfilename(initialdir=".")
            elif mode == "path":
                use_dir = messagebox.askyesno(
                    "输入类型", "是否选择目录作为预测输入？\n是=目录，否=单个CSV文件"
                )
                if use_dir:
                    p = filedialog.askdirectory(initialdir=".")
                else:
                    p = filedialog.askopenfilename(
                        initialdir=".", filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
                    )
            else:
                p = filedialog.asksaveasfilename(initialdir=".")
            if p:
                var.set(p)

        ttk.Button(parent, text="浏览", command=browse).grid(row=row, column=2, sticky="ew", pady=4)

    def _build_train_tab(self, tab: ttk.Frame) -> None:
        self._add_path_row(tab, 0, "valid目录", self.train_valid, "dir")
        self._add_path_row(tab, 1, "invalid目录", self.train_invalid, "dir")
        self._add_path_row(tab, 2, "模型文件", self.train_model, "save")
        self._add_path_row(tab, 3, "指标JSON", self.train_metrics, "save")
        self._add_path_row(tab, 4, "训练图表目录", self.train_chart_dir, "dir")

        ttk.Label(tab, text="测试集比例").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(tab, textvariable=self.test_size).grid(row=5, column=1, sticky="ew", pady=4)
        ttk.Label(tab, text="随机种子").grid(row=6, column=0, sticky="w", pady=4)
        ttk.Entry(tab, textvariable=self.random_state).grid(row=6, column=1, sticky="ew", pady=4)
        ttk.Label(tab, text="树数量").grid(row=7, column=0, sticky="w", pady=4)
        ttk.Entry(tab, textvariable=self.n_estimators).grid(row=7, column=1, sticky="ew", pady=4)
        ttk.Label(tab, text="并行线程数").grid(row=8, column=0, sticky="w", pady=4)
        ttk.Entry(tab, textvariable=self.n_jobs).grid(row=8, column=1, sticky="ew", pady=4)
        ttk.Label(tab, text="OOB步长").grid(row=9, column=0, sticky="w", pady=4)
        ttk.Entry(tab, textvariable=self.oob_step).grid(row=9, column=1, sticky="ew", pady=4)
        ttk.Checkbutton(tab, text="导出训练过程 OOB 曲线", variable=self.use_oob).grid(row=10, column=0, columnspan=3, sticky="w", pady=6)
        self.btn_train = ttk.Button(tab, text="开始训练并导图", command=self.start_train)
        self.btn_train.grid(row=11, column=0, columnspan=3, sticky="ew")

    def _build_pred_tab(self, tab: ttk.Frame) -> None:
        self._add_path_row(tab, 0, "模型文件", self.pred_model, "file")
        self._add_path_row(tab, 1, "输入(单CSV或目录)", self.pred_input, "path")
        self._add_path_row(tab, 2, "预测结果CSV", self.pred_csv, "save")
        self._add_path_row(tab, 3, "预测图表目录", self.pred_chart_dir, "dir")
        self.btn_pred = ttk.Button(tab, text="开始预测并导图", command=self.start_predict)
        self.btn_pred.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 0))

    def log(self, msg: str) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert(tk.END, f"[{now}] {msg}\n")
        self.log_box.see(tk.END)

    def log_async(self, msg: str) -> None:
        self.after(0, self.log, msg)

    def set_busy(self, b: bool, status: str) -> None:
        self.busy = b
        self.status.set(status)
        self.btn_train.configure(state=tk.DISABLED if b else tk.NORMAL)
        self.btn_pred.configure(state=tk.DISABLED if b else tk.NORMAL)

    def start_train(self) -> None:
        if self.busy:
            return
        self.set_busy(True, "训练中")
        self.log("开始训练...")
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self) -> None:
        try:
            valid_dir = Path(self.train_valid.get().strip())
            invalid_dir = Path(self.train_invalid.get().strip())
            model_out = Path(self.train_model.get().strip())
            metrics_out = Path(self.train_metrics.get().strip())
            chart_dir = Path(self.train_chart_dir.get().strip())
            test_size = float(self.test_size.get().strip())
            random_state = int(self.random_state.get().strip())
            n_estimators = int(self.n_estimators.get().strip())
            n_jobs = int(self.n_jobs.get().strip())

            self.log_async("读取数据并提取特征...")
            x, y, paths = build_dataset(valid_dir, invalid_dir)
            x_train, x_test, y_train, y_test, _, p_test = train_test_split(x, y, paths, test_size=test_size, random_state=random_state, stratify=y)

            self.log_async("训练模型...")
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs, class_weight="balanced")
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_prob = model.predict_proba(x_test)[:, 1]
            cm = confusion_matrix(y_test, y_pred)

            metrics = {
                "num_samples_total": int(len(y)),
                "num_train": int(len(y_train)),
                "num_test": int(len(y_test)),
                "class_distribution_total": {"valid": int((y == 0).sum()), "invalid": int((y == 1).sum())},
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision_invalid": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
                "recall_invalid": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
                "f1_invalid": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_prob)),
                "confusion_matrix": cm.tolist(),
                "classification_report": classification_report(y_test, y_pred, target_names=["valid", "invalid"], output_dict=True, zero_division=0),
            }

            model_out.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "label_map": {0: "valid", 1: "invalid"}}, model_out)
            metrics_out.parent.mkdir(parents=True, exist_ok=True)
            metrics_out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            pd.DataFrame({"file": [str(p) for p in p_test], "y_true": y_test, "y_pred": y_pred, "prob_invalid": y_prob}).to_csv(metrics_out.parent / "train_test_predictions.csv", index=False, encoding="utf-8-sig")

            chart_dir.mkdir(parents=True, exist_ok=True)
            charts = self._train_charts(chart_dir, y, y_test, y_pred, y_prob, cm, model, x_train, y_train, random_state, n_estimators)
            self.after(0, self._train_ok, metrics, charts, model_out, metrics_out)
        except Exception:
            self.after(0, self._worker_fail, "训练失败", traceback.format_exc())

    def _train_charts(self, chart_dir: Path, y: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, cm: np.ndarray, model: RandomForestClassifier, x_train: np.ndarray, y_train: np.ndarray, random_state: int, n_estimators: int) -> dict[str, Path]:
        charts: dict[str, Path] = {}

        counts = [int((y == 0).sum()), int((y == 1).sum())]
        fig = Figure(figsize=(6, 4), dpi=120); ax = fig.add_subplot(111)
        ax.bar(["valid", "invalid"], counts, color=["#4C78A8", "#E45756"]); ax.set_title("类别分布"); ax.set_ylabel("count")
        p = chart_dir / "01_class_distribution.png"; fig.tight_layout(); fig.savefig(p, dpi=220); charts["训练-类别分布"] = p

        fig = Figure(figsize=(5, 4), dpi=120); ax = fig.add_subplot(111)
        im = ax.imshow(cm, cmap="Blues"); ax.set_title("混淆矩阵")
        ax.set_xticks([0, 1], labels=["pred_v", "pred_i"]); ax.set_yticks([0, 1], labels=["true_v", "true_i"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        p = chart_dir / "02_confusion_matrix.png"; fig.tight_layout(); fig.savefig(p, dpi=220); charts["训练-混淆矩阵"] = p

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig = Figure(figsize=(5, 4), dpi=120); ax = fig.add_subplot(111)
        ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_prob):.4f}"); ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_title("ROC"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(loc="lower right")
        p = chart_dir / "03_roc_curve.png"; fig.tight_layout(); fig.savefig(p, dpi=220); charts["训练-ROC"] = p

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        fig = Figure(figsize=(5, 4), dpi=120); ax = fig.add_subplot(111)
        ax.plot(recall, precision, color="#F58518"); ax.set_title("PR曲线"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        p = chart_dir / "04_pr_curve.png"; fig.tight_layout(); fig.savefig(p, dpi=220); charts["训练-PR"] = p

        top = np.argsort(model.feature_importances_)[::-1][:15]
        fig = Figure(figsize=(8, 5), dpi=120); ax = fig.add_subplot(111)
        names = [FEATURE_NAMES[i] for i in top][::-1]; vals = model.feature_importances_[top][::-1]
        ax.barh(np.arange(len(names)), vals, color="#54A24B"); ax.set_yticks(np.arange(len(names)), labels=names); ax.set_title("特征重要性Top15")
        p = chart_dir / "05_feature_importance.png"; fig.tight_layout(); fig.savefig(p, dpi=220); charts["训练-特征重要性"] = p

        fig = Figure(figsize=(6, 4), dpi=120); ax = fig.add_subplot(111)
        ax.hist(y_prob[y_test == 0], bins=25, alpha=0.6, label="true_valid")
        ax.hist(y_prob[y_test == 1], bins=25, alpha=0.6, label="true_invalid")
        ax.set_title("测试集概率分布"); ax.set_xlabel("prob_invalid"); ax.legend()
        p = chart_dir / "06_prob_distribution_test.png"; fig.tight_layout(); fig.savefig(p, dpi=220); charts["训练-概率分布"] = p

        if self.use_oob.get():
            step = max(1, int(self.oob_step.get().strip()))
            points, errs = [], []
            rf = RandomForestClassifier(n_estimators=step, warm_start=True, oob_score=True, bootstrap=True, class_weight="balanced", random_state=random_state, n_jobs=1)
            self.log_async("计算OOB训练过程曲线...")
            for n in range(step, n_estimators + 1, step):
                rf.set_params(n_estimators=n); rf.fit(x_train, y_train)
                points.append(n); errs.append(1.0 - float(rf.oob_score_))
            fig = Figure(figsize=(6, 4), dpi=120); ax = fig.add_subplot(111)
            ax.plot(points, errs, marker="o", color="#B279A2"); ax.set_title("OOB误差-训练过程"); ax.set_xlabel("n_estimators"); ax.set_ylabel("oob_error")
            p = chart_dir / "07_oob_curve.png"; fig.tight_layout(); fig.savefig(p, dpi=220); charts["训练-OOB过程曲线"] = p

        return charts

    def _train_ok(self, metrics: dict, charts: dict[str, Path], model_out: Path, metrics_out: Path) -> None:
        self.log(f"训练完成 acc={metrics['accuracy']:.4f}, f1_invalid={metrics['f1_invalid']:.4f}, recall_invalid={metrics['recall_invalid']:.4f}, auc={metrics['roc_auc']:.4f}")
        self.log(f"模型: {model_out}")
        self.log(f"指标: {metrics_out}")
        self._set_charts(charts)
        self.set_busy(False, "训练完成")
        messagebox.showinfo("训练完成", "模型与图表已导出")

    def start_predict(self) -> None:
        if self.busy:
            return
        self.set_busy(True, "预测中")
        self.log("开始预测...")
        threading.Thread(target=self._predict_worker, daemon=True).start()

    def _predict_worker(self) -> None:
        try:
            model_path = Path(self.pred_model.get().strip())
            input_text = self.pred_input.get().strip()
            input_path = Path(input_text)
            if not input_path.exists():
                raise ValueError(f"输入路径不存在: {input_path}")
            output_csv = Path(self.pred_csv.get().strip())
            chart_dir = Path(self.pred_chart_dir.get().strip())

            artifact = joblib.load(model_path)
            model = artifact["model"]
            files = [input_path] if input_path.is_file() else iter_csv_files(input_path)
            if not files:
                raise ValueError("目录内无CSV")

            idx_mean = FEATURE_NAMES.index("mu_mean")
            idx_std = FEATURE_NAMES.index("mu_std")
            rows = []
            for i, f in enumerate(files, 1):
                x = extract_features_from_csv(f).reshape(1, -1)
                pred = int(model.predict(x)[0])
                prob = float(model.predict_proba(x)[0, 1])
                rows.append({"file": str(f), "pred_label": "invalid" if pred == 1 else "valid", "prob_invalid": prob, "mu_mean": float(x[0, idx_mean]), "mu_std": float(x[0, idx_std])})
                if i % 300 == 0:
                    self.log_async(f"预测进度 {i}/{len(files)}")

            df = pd.DataFrame(rows).sort_values("prob_invalid", ascending=False)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            chart_dir.mkdir(parents=True, exist_ok=True)
            charts = self._pred_charts(chart_dir, df)
            self.after(0, self._pred_ok, df, output_csv, charts)
        except Exception:
            self.after(0, self._worker_fail, "预测失败", traceback.format_exc())

    def _pred_charts(self, chart_dir: Path, df: pd.DataFrame) -> dict[str, Path]:
        charts: dict[str, Path] = {}

        cnt = df["pred_label"].value_counts().reindex(["valid", "invalid"], fill_value=0)
        fig = Figure(figsize=(5, 4), dpi=120); ax = fig.add_subplot(111)
        ax.bar(["valid", "invalid"], cnt.values, color=["#4C78A8", "#E45756"]); ax.set_title("预测类别计数")
        p = chart_dir / "01_pred_label_count.png"; fig.tight_layout(); fig.savefig(p, dpi=220); charts["预测-类别计数"] = p

        fig = Figure(figsize=(6, 4), dpi=120); ax = fig.add_subplot(111)
        ax.hist(df["prob_invalid"], bins=30, color="#F58518", alpha=0.85); ax.set_title("预测概率分布"); ax.set_xlabel("prob_invalid")
        p = chart_dir / "02_pred_prob_hist.png"; fig.tight_layout(); fig.savefig(p, dpi=220); charts["预测-概率分布"] = p

        top = df.head(min(20, len(df))).copy(); top["name"] = top["file"].map(lambda x: Path(x).name)
        fig = Figure(figsize=(9, 5), dpi=120); ax = fig.add_subplot(111)
        ax.barh(np.arange(len(top))[::-1], top["prob_invalid"][::-1], color="#E45756")
        ax.set_yticks(np.arange(len(top))[::-1], labels=top["name"][::-1]); ax.set_title("高风险Top20")
        p = chart_dir / "03_top_risk.png"; fig.tight_layout(); fig.savefig(p, dpi=220); charts["预测-高风险Top20"] = p

        fig = Figure(figsize=(6, 4), dpi=120); ax = fig.add_subplot(111)
        color = np.where(df["pred_label"] == "invalid", "#E45756", "#4C78A8")
        ax.scatter(df["mu_mean"], df["mu_std"], c=color, s=14, alpha=0.75)
        ax.set_title("mu_mean-mu_std散点"); ax.set_xlabel("mu_mean"); ax.set_ylabel("mu_std")
        p = chart_dir / "04_scatter_mean_std.png"; fig.tight_layout(); fig.savefig(p, dpi=220); charts["预测-特征散点"] = p

        sorted_prob = np.sort(df["prob_invalid"].to_numpy())[::-1]
        fig = Figure(figsize=(6, 4), dpi=120); ax = fig.add_subplot(111)
        ax.plot(sorted_prob, color="#72B7B2"); ax.set_title("风险排序曲线"); ax.set_xlabel("rank"); ax.set_ylabel("prob_invalid")
        p = chart_dir / "05_risk_rank_curve.png"; fig.tight_layout(); fig.savefig(p, dpi=220); charts["预测-风险排序曲线"] = p

        top_file = Path(str(df.iloc[0]["file"]))
        try:
            t, mu = read_signal(top_file)
            fig = Figure(figsize=(9, 4), dpi=120); ax = fig.add_subplot(111)
            ax.plot(t, mu, color="#4C78A8", linewidth=1.5)
            ax.set_title(f"高风险样本信号: {top_file.name} | pred={df.iloc[0]['pred_label']} | p={df.iloc[0]['prob_invalid']:.4f}")
            ax.set_xlabel("t_s"); ax.set_ylabel("mu_true")
            p = chart_dir / "06_top_risk_signal.png"; fig.tight_layout(); fig.savefig(p, dpi=220); charts["预测-高风险信号"] = p
        except Exception:
            pass

        return charts

    def _pred_ok(self, df: pd.DataFrame, output_csv: Path, charts: dict[str, Path]) -> None:
        invalid_n = int((df["pred_label"] == "invalid").sum())
        self.log(f"预测完成 total={len(df)}, invalid={invalid_n}, valid={len(df)-invalid_n}")
        self.log(f"预测CSV: {output_csv}")
        self._set_charts(charts)
        self.set_busy(False, "预测完成")
        messagebox.showinfo("预测完成", "预测结果和图表已导出")

    def _worker_fail(self, title: str, detail: str) -> None:
        self.set_busy(False, title)
        self.log(detail)
        messagebox.showerror(title, detail)

    def _set_charts(self, charts: dict[str, Path]) -> None:
        self.chart_files = charts
        keys = list(charts.keys())
        self.combo.configure(values=keys)
        if keys:
            self.chart_key.set(keys[0])
            self.show_chart()

    def show_chart(self) -> None:
        key = self.chart_key.get().strip()
        if not key or key not in self.chart_files:
            return
        p = self.chart_files[key]
        if not p.exists():
            self.log(f"图表不存在: {p}")
            return
        img = mpimg.imread(p)
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(p.name)
        self.canvas.draw_idle()

    def open_chart_dir(self) -> None:
        key = self.chart_key.get().strip()
        if key and key in self.chart_files:
            d = self.chart_files[key].parent
        else:
            d = Path(".")
        try:
            os.startfile(str(d))
        except Exception:
            self.log(f"无法打开目录: {d}")


if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except tk.TclError as e:
        print("GUI启动失败: 当前Python环境缺少可用的Tcl/Tk运行时。")
        print("请安装带Tkinter组件的Python，或修复TCL_LIBRARY/TK_LIBRARY环境变量。")
        print(f"原始错误: {e}")
