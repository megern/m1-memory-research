# دليل التشغيل المحلي

هذه دراسة على Apple M1 بذاكرة موحدة 16 GiB. لا تتضمن تشغيلًا أو تدريبًا على خدمات سحابية. تحتاج الشبكة لتنزيل البرامج والأوزان العامة؛ بعد ذلك تنفذ التجارب من الملفات المحلية.

## ما الذي ندربه؟

نستخدم نموذج Qwen3 صغيرًا جاهزًا ومضغوطًا، ونحدث محوّل LoRA لكل مهمة. لا ندرّب نموذجًا تأسيسيًا من الصفر، ولا ندرب نموذج 70B على هذا الجهاز. محوّل المهمة يحتاج النموذج الأساسي عند التشغيل؛ حجم ملف المحوّل وحده ليس حجم النظام الكامل.

المهمتان الأصليتان هما تصنيف أولوية سجل دخول، وتصنيف خصائص سياسة سير عمل GitHub. البيانات اصطناعية والقواعد مكتوبة في التعليمات. النسخة العددية تحتفظ بأخطائها، ومنها مثال دخول أجاب عنه المحوّل المرجعي بشكل خاطئ.

النسخة `normalized` تنفذ المقارنات العددية بالكود أولًا، ثم تمرر ثلاث حقائق منطقية للنموذج. استخدم معها `--normalize`. حذف هذا الخيار يغيّر المدخلات التي اختبرناها. القواعد المباشرة تستطيع حل هذه المهمة كاملة دون نموذج، لذلك لا نستنتج من نتيجتها قدرة أمنية عامة أو قدرة النموذج على الحساب.

## التحضير

أنشئ بيئة Python محلية على Apple Silicon وثبت `requirements.txt`. من مجلد المشروع، نزّل النموذج الصغير صراحة:

```sh
python3 tools/download_base.py --output /absolute/local/qwen3-0.6b-4bit
```

استبدل المسار بمجلد محلي جديد. البرنامج يثبت نسخة الأوزان ويتحقق من بصماتها. لا تضع مفاتيح وصول أو ملفات خاصة داخل مستودع الدراسة.

## إعادة التدريب

```sh
python3 tools/run_experiments.py --model /absolute/local/qwen3-0.6b-4bit --series confirmatory --output runs/new-confirmatory
python3 tools/run_experiments.py --model /absolute/local/qwen3-0.6b-4bit --series normalized --output runs/new-normalized
```

كل أمر يدرب البذور الثلاث بالتتابع ويختبرها مع النموذج الأصلي. اختر مجلد نتائج جديدًا؛ السجلات المنشورة تبقى محفوظة. التجربة الأكبر تستخدم `--series size1p7` بعد تنزيل النموذج المحدد في `models/qwen3-1.7b.json`.

## تجربة مدخل واحد

```sh
python3 tools/predict.py --model /absolute/local/qwen3-0.6b-4bit --adapter /absolute/local/workflow-adapter --task workflow --record examples/workflow.json --language ar
python3 tools/predict.py --model /absolute/local/qwen3-0.6b-4bit --adapter /absolute/local/normalized-adapter --task auth --normalize --record examples/auth.json --language ar
```

تظهر الإجابة الخام وصحة تنسيقها. لا يُستبدل تصنيف خاطئ بالإجابة الصحيحة التي نعرفها من القاعدة. هذه أداة تجريبية وليست نظامًا لاتخاذ قرارات أمنية تشغيلية.

## تشغيل 70B

```sh
python3 tools/install_runtime.py --output runtime
python3 tools/download_model.py --manifest models/llama70b.json --output /absolute/local/models/llama70b.gguf --workers 4
python3 tools/run_large_model.py --runtime runtime/llama-b11457/llama-completion --model /absolute/local/models/llama70b.gguf --manifest models/llama70b.json --output results/new-70b-run --timeout 900 --tokens 8
```

حجم الملف المختار نحو 19.1 GB، ويتجاوز الذاكرة الفعلية. التجربة تستخدم المعالج والقرص عبر mmap، وتقلل طول السياق وعدد الرموز. توجد حدود توقف لضغط الذاكرة وزيادة swap والوقت؛ لا تتوقف برامج المستخدم الأخرى. تحميل الملف لا يثبت نجاح الاستدلال، والإجابة القصيرة لا تثبت سرعة محادثة عملية.

اقرأ `report.json` والمخرجات الخام والتوقيت قبل الحكم على النتيجة. بصمة الملف تُحسب قبل قياس زمن التنفيذ وقد تسخن ذاكرة الملفات؛ لا نصف التجربة بأنها اختبار قرص بذاكرة مؤقتة باردة. ذاكرة MLX وذاكرة العملية والذاكرة الموحدة ليست ثلاث مجموعات مستقلة يمكن جمعها.

## قراءة النتائج والأوراق

تعرض `results` كل بذور التدريب، النموذج الأصلي، الإجابات الخام، وبصمات البيانات والأوزان. ملفات `papers` مسودات علمية قابلة للتحرير، وليست أوراقًا محكمة. نميز بين صحة JSON وصحة التصنيف، ونحتفظ بالمقارنات والتجارب السلبية. تُعرض الأرقام المكتملة في README والأوراق بعد تسجيل التجربة الفعلية.
