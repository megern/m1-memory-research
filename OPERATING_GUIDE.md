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

## تشغيل الأوزان الأصلية دون تكميم

هذه الأداة الجديدة تستخدم Qwen3-0.6B الأصلي بدقة BF16، وتبقي الأوزان دون تكميم. النموذج صغير ويسع ذاكرة الجهاز؛ لا يمثل هذا تشغيل 70B بالأوزان الأصلية. اختر مجلد نموذج، ويجب أن يكون مجلد الطبقات بجواره بالاسم نفسه مع اللاحقة `-layers`.

```sh
python3 tools/download_base.py --manifest models/qwen3-0.6b-original.json --output /absolute/local/qwen3-original
python3 tools/shard_original_layers.py --model /absolute/local/qwen3-original --output /absolute/local/qwen3-original-layers
python3 tools/local_original_inference.py --model /absolute/local/qwen3-original --budget-mib 640 --tokens 64 --prompt "Return only Python code for a function add(a, b) that returns a + b." --output results/new-original-demo
```

تنزيل الأوزان علني، ويجري الحساب على الجهاز فقط. التقسيم ينسخ الأوزان إلى ملفات إضافية ويتحقق من بقاء بياناتها؛ يحتاج مساحة إضافية تقارب حجم النموذج. هذه الطريقة في تجهيز الملفات لا تصلح تلقائيًا لنموذج 70B على المساحة الحالية؛ الوصول المباشر إلى ملفات النموذج الأصلية مطلوب قبل التوسع.

الميزانية `640` تخص تقدير تخصيص MLX وفحوص الذاكرة النشطة أثناء الحساب، ولا تحد إجمالي RAM أو ذاكرة النظام المؤقتة. اختير تخزين سبع طبقات بهذا الإعداد، وبلغت الذروة المقاسة نحو 605MiB في المقارنات. قد يتوقف سؤال أطول إذا تجاوز الحدود؛ لا تضمن المعادلة وحدها ملاءمة كل سياق. الأداة تحد السياق إلى 256 رمزًا، والسؤال إلى 128 رمزًا بعد القالب، والإجابة إلى 64 رمزًا. تعرض الإجابة وبصمتها التشغيلية وحد التوقف في `report.json`؛ جودة الإجابة تعتمد على النموذج الأصلي.

## إعادة مقارنة طرق التحميل

```sh
python3 tools/run_original_precision.py --model /absolute/local/qwen3-original --experiment scheduler-reference --output results/new-scheduler-reference
python3 tools/run_scheduler_trials.py --model /absolute/local/qwen3-original --output results/new-scheduler-trials
python3 tools/run_budgeted_trials.py --model /absolute/local/qwen3-original --reference --output results/new-budgeted-reference
python3 tools/run_budgeted_trials.py --model /absolute/local/qwen3-original --output results/new-budgeted-trials
```

الأمر الأول يولّد مرجع الحساب إلى مجلد بجوار النموذج بلاحقة `-scheduler-reference`، والثالث بلاحقة `-budgeted-reference`. المراجع ونتائج التجارب لا تكتب فوق ملفات قائمة. استخدم مجلد نموذج جديدًا عند إعادة توليد مرجع موجود. الأوامر تقارن القيم الرقمية والتوكنات على المدخلات نفسها، وتحتفظ بكل الجولات وترتيبها العشوائي وبصمات المصدر؛ لا تختار أسرع جولة وحدها.

المقارنة الأولى: خمس طرق، خمس جولات، أربع حالات قصيرة. المتابعة الاستكشافية: أربع طرق، ثلاث جولات، ست حالات إضافية. ملف البروتوكول يثبت الأسئلة والسياق العددي وحد التوليد. توقيت المقارنة يستثني تحميل النموذج الأولي والتحقق من البصمات، بينما تقرير العملية يحفظ زمنها الكلي. حالة ذاكرة الملفات غير متحكم بها وتسخن بقراءة البصمات؛ ليست هذه تجربة قرص بارد.

لإعادة رسم النتائج، ثبّت متطلبات الرسوم الموجودة ثم استخدم:

```sh
python3 tools/plot_original_scheduler.py --summary results/new-budgeted-trials/summary.json --output figures/new-budgeted-engine
```

النتيجة المتكررة حتى الآن: تخزين بعض الطبقات خفّض وسيط زمن الحساب بنحو 5% مقابل التحميل المتتابع، مع استهلاك MLX أعلى منه. التشغيل المقيم أسرع لكنه يستخدم تخصيص MLX أعلى. اقرأ الفروق والقيود في `results/original-scheduling-development-notes.txt` قبل نقل الأرقام إلى منشور أو ورقة.

## الوصول المباشر إلى الأوزان الأصلية

هذه المتابعة تلغي الحاجة إلى نسخ الطبقات. تقرأ طبقة واحدة من الملفات الأصلية، وصفوف تضمين السؤال المطلوبة، وتحسب درجات جميع مفردات النموذج على أجزاء صغيرة دون تكميم الأوزان. أثبت اختبار النموذج الصغير تطابق 48 رمزًا مع المرجع، مع فروق عددية في ست خطوات بلغت 0.0625 كحد أقصى. بلغ تخصيص MLX نحو 61 MiB وذاكرة العملية المرصودة نحو 276 MiB؛ هذا ليس إجمالي استهلاك RAM.

لتجربة النموذج الصغير دون ملفات طبقات إضافية:

```sh
python3 tools/run_direct_original.py --model /absolute/local/qwen3-original --manifest models/qwen3-0.6b-original.json --output results/new-direct-small --budget-mib 256 --head-rows 2048 --tokens 8 --timeout 180
```

لإعادة الفحص العددي، أضف `--reference /absolute/local/qwen3-original-budgeted-reference` بعد إنشاء المرجع بالأمر السابق. يحتفظ التقرير بالتطابق العددي منفصلًا عن اكتمال التنفيذ؛ فشل التطابق أو التنفيذ يعطي رمز خروج غير صفري. لا تخلط فحص القيم العددية بتقييم جودة الإجابة.

اكتمل تنزيل Qwen3-14B الأصلي بحجم نحو 29.5 GB والتحقق من أوزانه، وهي تتجاوز ذاكرة الجهاز. توقفت المحاولة الأولى بسبب زيادة swap، ثم اكتمل تكرار واحد بالإعدادات نفسها وأجاب Paris و323 حتى نهاية الإجابتين. هذا اختبار قصير لإمكانية التشغيل، وليس إثبات استقرار أو جودة عامة أو تشغيل 70B الأصلي. للتنزيل والاستئناف:

```sh
python3 tools/acquire_original_model.py --manifest models/qwen3-14b-original.json --output /absolute/local/qwen3-14b-original
```

بعد اكتماله، نفّذ الاختبار الموثق في `models/direct-original-14b-protocol.json`:

```sh
python3 tools/run_direct_original.py --model /absolute/local/qwen3-14b-original --manifest models/qwen3-14b-original.json --output results/new-direct-14b --budget-mib 1024 --head-rows 2048 --tokens 4 --timeout 900 --prompt "Reply with exactly the single English word naming the capital of France." --prompt "What is 17 multiplied by 19? Give only the number."
```

يمكن تشغيل المتابعة مرة واحدة أثناء التنزيل في نافذة محلية أخرى:

```sh
python3 tools/wait_original_probe.py --model /absolute/local/qwen3-14b-original --manifest models/qwen3-14b-original.json --protocol models/direct-original-14b-protocol.json --output /absolute/local/fresh-continuation
```

تحفظ المتابعة حالة الانتظار، ونسخة المصدر، والبروتوكول، ثم نتائج الاختبار الأول. لا تشغّل الاختبار اليدوي والمتابعة معًا. تبقى الأوزان على جهازك ولا ترفع إلى المستودع. حد 1024 يخص تخصيص MLX النشط؛ تراقب الأداة أيضًا ذاكرة العملية وزيادة swap والوقت، وتوقف عملياتها وحدها عند تجاوز الحدود. يجب أن يبقى الجهاز مستيقظًا ومتصلًا بالشبكة أثناء التنزيل؛ لا تغير الأداة إعدادات الطاقة. لا تستخدم ملفات نتائج قائمة عند إعادة التشغيل.

نتائج 14B محفوظة في `results/direct-original-14b-first-v1` و`results/direct-original-14b-repeat-v1`. استغرق التكرار المكتمل نحو 75 ثانية بعد نحو 17 ثانية للتحقق من الملفات. ذروة MLX كانت 1.153 GiB، فتجاوزت الذروة العابرة إعداد الفحص النشط 1 GiB؛ لا تستخدم الإعداد بوصفه حدًا صارمًا للذروة أو لإجمالي RAM. ذروة ذاكرة العملية المرصودة نحو 0.924 GiB، وزيادة swap للنظام بلغت 473 MiB. استهلاك ذاكرة الملفات والنظام إضافي، ولا تجمع RSS مع MLX. نحتفظ بالمحاولة المتوقفة والتكرار المكتمل ولا نستنتج استقرار التشغيل من نجاح واحد.

## مقارنة القراءة وحماية مرحلة التجهيز

المتابعة الجديدة قارنت ثلاث طرق قراءة على 14B في جولتين بترتيب عشوائي، وثلاثة أسئلة قصيرة جديدة. اكتملت خمس محاولات من أصل ست. التشغيل المعتاد عبر MLX أكمل جولتيه، وكانت أزمنة الحساب نحو 119 و122 ثانية. القراءة المباشرة العادية كانت أبطأ في جولتيها، والقراءة بطلب عدم التخزين المؤقت أكملت محاولة وتوقفت أخرى. جميع المقارنات المكتملة مع الطريقة المعتادة طابقت تسلسل الرموز. هذه نتائج استكشافية قليلة؛ لا نعامل المحاولة المتوقفة التي استغرقت خمس ثوانٍ على أنها أسرع.

توجد مشكلة أخرى موثقة: زاد swap للنظام بنحو 1035 MiB أثناء قراءة البصمات قبل مراقبة الاستدلال في إحدى المحاولات. المراقبة القديمة تبدأ بعد التحقق؛ أما الأداة الحالية فتراقب التحقق نفسه وتحافظ على مرجع swap واحد منذ البداية حتى انتهاء التوليد. إذا توقف التحقق، يوضح التقرير `model_started=false` ولا يبدأ تشغيل النموذج. تخص الزيادة النظام كله وتشمل التطبيقات الأخرى؛ لا ننسب مصدرها إلى النموذج وحده.

يمكن تجربة سؤال جديد بالحماية الحالية:

```sh
python3 tools/run_direct_original.py --model /absolute/local/qwen3-14b-original --manifest models/qwen3-14b-original.json --output results/new-guarded-code --io-mode native --verification-nocache --budget-mib 1024 --head-rows 2048 --tokens 16 --timeout 900 --prompt "Reply with only the Python expression that reverses a list named items without changing it."
```

الخيار `--verification-nocache` يطلب من macOS عدم تخزين بيانات ملفات التحقق مؤقتًا في مقابض القراءة التي تفتحها الأداة وحدها. لا يغير ذاكرة الجهاز المؤقتة عالميًا ولا يثبت غياب كل الصفحات المخزنة. القراءة المعتادة تبقى الإعداد الافتراضي للأوزان لأنها كانت الأسرع في المقارنة. يمكن اختيار `--io-mode raw-cached` أو `raw-nocache` للفحص البحثي؛ لم نثبت تحسن سرعتها أو استقرارها.

أنتج الاختبار البرمجي الأول التعبير الصحيح داخل علامتي Markdown، وانتهى طبيعيًا. فشل تحليل النص الخام كتعبير Python بسبب العلامتين، ثم نجح فحص بنية التعبير بعد حذف الغلاف الخارجي كاملًا في معالجة عرض لاحقة موثقة. لم ننفذ الكود الذي ولده النموذج. استغرق التشغيل نحو 83 ثانية والتحقق نحو 25 ثانية؛ ذروة MLX نحو 0.669 GiB، والزيادة المرصودة القصوى في swap منذ بداية التجهيز نحو 23 MiB. السؤال والإعدادات مختلفة عن مقارنة القراءة، لذلك لا نستنتج منها تحسنًا عامًا في استهلاك الذاكرة.

لإعادة المقارنة التاريخية بالإعدادات القديمة نفسها، استخدم نسختها المجمدة:

```sh
python3 results/original-io-trials-v1/source/run_io_trials.py --model /absolute/local/qwen3-14b-original --manifest models/qwen3-14b-original.json --protocol models/original-io-protocol.json --output results/new-io-reproduction
python3 tools/analyze_io_trials.py --results results/new-io-reproduction
python3 tools/plot_io_trials.py --summary results/new-io-reproduction/summary.json --output figures/new-io-reproduction
```

تحتفظ النتائج بكل المحاولات ونسخ المصدر والبروتوكول وبصماتها. المراقبة الحالية أشد وتشمل التجهيز؛ النسخة المجمدة مطلوبة لإعادة الدراسة التاريخية. اقرأ مسودة `papers/original-io-study.txt` والتحليل في `results/original-io-trials-v1/descriptive-analysis.json`. المسودة ليست ورقة محكمة. لا تشمل هذه المرحلة تدريب نموذج جديد أو تشغيل 70B بالأوزان الأصلية.

## معالجة السؤال على أجزاء مع الأوزان الأصلية

الخيار الافتراضي يعالج السؤال كاملًا، بحد 128 رمزًا للمدخل و256 رمزًا للسياق الكلي. التقسيم يغيّر أحجام الحساب بدقة BF16؛ لا نفترض أن درجات الكلمات أو الإجابات ستطابق المعالجة دفعة واحدة. فشل إعداد 8 رموز في تطابق الدرجات مع المرجع الأصلي، رغم تطابقه مع مكتبة MLX عند استخدام التقسيم نفسه.

الخيار `chunk-major` يمرّر كل جزء عبر جميع الطبقات، ويعيد تحميلها للجزء التالي. الخيار `layer-major` يحتفظ بأوزان طبقة واحدة أثناء معالجة أجزاء السؤال كلها، ثم يجمع مخرجاتها وينتقل إلى الطبقة التالية. يحافظ على مواضع ذاكرة الانتباه والقناع السببي لكل طبقة، ولا يحذف طبقات أو درجات مفردات. لا يغير دقة الأوزان، ولا يدرب نموذجًا جديدًا.

لتجربة الترتيب الثاني محليًا على ملفات النموذج الأصلي التي نزلتها مسبقًا:

```sh
python tools/run_direct_original.py \
  --model /absolute/local/qwen3-14b-original \
  --manifest models/qwen3-14b-original.json \
  --output /absolute/local/new-layer-major-result \
  --prompt 'Reply with exactly the capital of France.' \
  --tokens 4 --prefill-chunk-size 32 --prefill-schedule layer-major \
  --verification-nocache
```

مجلد النتائج يجب أن يكون جديدًا. البرنامج يتحقق من جميع الملفات الأصلية قبل التشغيل ويراقب الذاكرة خلال التحقق والتشغيل. الحد 1024 ميبيبايت هو فحص تخصيص MLX النشط، وليس حدًا صارمًا لذاكرة الجهاز. القراءة من القرص وبصمات الملفات قد تستفيدان من ذاكرة نظام التشغيل؛ لا نصفها بأنها قراءة من قرص بارد.

لإعادة دراسة ترتيب الطبقات بدقة استخدم نسخة المصدر المثبتة مع نتائجها:

```sh
python results/original-prefill-schedule-trials-v1/source/run_prefill_trials.py \
  --model /absolute/local/qwen3-14b-original \
  --manifest results/original-prefill-schedule-trials-v1/manifest.json \
  --protocol results/original-prefill-schedule-trials-v1/protocol.json \
  --output /absolute/local/new-schedule-study
```

هذه دراسة هندسية قصيرة، لا تثبت جودة أمنية عامة أو اختراعًا جديدًا. نموذج 70 مليار بأوزانه الأصلية وتدريب النماذج الكبيرة ليسا ضمن نتائج هذه المرحلة.

## التجارب الأطول واستخدام المعالج وGPU

يبقى حد المدخل الافتراضي 128 رمزًا والسياق 256 رمزًا. يمكن اختيار حدود تجريبية تصل إلى 1024 رمزًا للمدخل و2048 للسياق. اختبارنا الجديد يستخدم سؤالين من 471 و891 رمزًا؛ لا يعني ذلك اختبار جميع الأطوال حتى 2048.

```sh
python tools/run_direct_original.py \
  --model /absolute/local/qwen3-14b-original \
  --manifest models/qwen3-14b-original.json \
  --output /absolute/local/new-long-result \
  --prompt 'ضع هنا السؤال الذي تريد تجربته.' \
  --tokens 4 --budget-mib 2048 \
  --max-input-tokens 1024 --max-context-tokens 2048 \
  --prefill-chunk-size 128 --prefill-schedule layer-major \
  --verification-nocache --verification-workers 8
```

يحسب النموذج على GPU الماك، ويمكن لثمانية عمال من المعالج التحقق من ملفات الأوزان بالتوازي قبل بدء الحساب. يبقى الافتراضي عامل تحقق واحدًا. لا تعني هذه الإعدادات إشغال كل الأنوية بنسبة 100% طوال الوقت؛ البرنامج يسجل جهاز الحساب ووقت المعالج وعدد خيوطه، ولا يقيس نسبة إشغال GPU. استخدام الذاكرة الإضافية لخدمة السؤال هو الهدف، وليس ملء الرام دون فائدة.

تظل حواجز الإيقاف فعالة حتى مع ميزانية MLX الأكبر: نمو swap المرصود فوق 512 ميبيبايت، أو تجاوز حد RSS، أو استمرار انخفاض الذاكرة المتاحة، أو انتهاء الوقت. لا يغلق البرنامج التطبيقات الأخرى ولا يغير إعدادات النظام. المحاولات المتوقفة تبقى ضمن الدراسة؛ الذاكرة الافتراضية على مستوى الجهاز تشمل التطبيقات الأخرى.

لإعادة الدراسة المثبتة:

```sh
python results/original-long-context-trials-v1/source/run_prefill_trials.py \
  --model /absolute/local/qwen3-14b-original \
  --manifest results/original-long-context-trials-v1/manifest.json \
  --protocol results/original-long-context-trials-v1/protocol.json \
  --output /absolute/local/new-long-study
```

ولمقارنة التحقق بعامل واحد وثمانية عمال دون تحميل النموذج:

```sh
python tools/benchmark_verification.py \
  --model /absolute/local/qwen3-14b-original \
  --manifest models/qwen3-14b-original.json \
  --output /absolute/local/new-hash-study
```

تطبق مقارنة التحقق تلميح القراءة نفسه على الملفات وتحافظ على البصمات والحواجز، لكن ذاكرة نظام التشغيل ليست مضبوطة؛ ليست هذه مقارنة قرص بارد.


## قراءة أوزان الطبقة التالية أثناء الحساب (تجريبية)

أضف `--io-mode raw-cached --prefetch-layers 1` إلى مشغّل
`run_direct_original.py` لتشغيل قارئ واحد لأوزان الطبقة التالية. تظل حسابات
MLX كلها في المسار الرئيسي؛ القارئ يعيد بايتات BF16 الأصلية بعد فحص حدود
الملف وهويته، ولا يضغط الأوزان أو يغيرها. الافتراضي `--prefetch-layers 0`.
لا يمكن الجمع بين هذه الميزة و`--io-mode native`.

هذه الميزة قد تزيد RSS بسبب تخزين بايتات طبقة إضافية، نحو 630 ميبيبايت
في نموذج 14B. ميزانية MLX لا تشمل هذه البايتات ولا تعني حدًا للرام الكلي.
الاختبارات الرقمية على 0.6B وحدها لا تثبت جودة 14B أو استقرار التشغيل.
تظل حواجز الضغط السابقة فعالة؛ لا ترفعها لتجاوز نتائج فاشلة.

لإعادة مقارنة الطرق الثلاث باستخدام الشفرة المثبتة:

```sh
python results/original-read-ahead-trials-v1/source/run_prefill_trials.py \
  --model /absolute/local/qwen3-14b-original \
  --manifest results/original-read-ahead-trials-v1/manifest.json \
  --protocol results/original-read-ahead-trials-v1/protocol.json \
  --output /absolute/local/new-read-ahead-study
```

يعرض التقرير كل المحاولات، حتى المتوقفة. وقت المحاولة المتوقفة لا يمثل
وقت إنجاز السؤال، وذاكرة swap على مستوى الجهاز تشمل التطبيقات الأخرى.


## القراءة الجزئية المحدودة وتجربة التشغيل المتتالي

الاختيار الجديد `--prefetch-prefix-mib 64` يحتفظ بمقدمة لا تتجاوز
64 ميبيبايت من الطبقة التالية عند استعمال `--prefetch-layers 1`. تُقرأ
بقية الأوزان وتُحوّل إلى مصفوفات BF16 في المسار الرئيسي، موترًا واحدًا
في كل مرة. استعمل `--io-mode raw-cached`. الجمع مع
`--prefetch-layers 0` يعطي المقارنة المتسلسلة المطابقة دون قارئ خلفي.
القيمة صفر لمقدمة القراءة تبقي طريقة الطبقة الكاملة السابقة.

هذا حد لمقدمة القراءة فقط؛ ذاكرة الأوزان الحالية ونسخ البناء وKV
والبرنامج لا تدخل في هذا الحد. لا تفترض أن استهلاك البرنامج 64 ميبيبايت
أو أن القراءة الجزئية أسرع. الافتراضي لا يزال التحميل الأصلي `native`.

لإعادة تجربة 14B المثبتة، اختر مجلد نتائج جديدًا:

```sh
python results/original-prefix-read-ahead-trials-v1/source/run_prefill_trials.py \
  --model /absolute/local/qwen3-14b-original \
  --manifest results/original-prefix-read-ahead-trials-v1/manifest.json \
  --protocol results/original-prefix-read-ahead-trials-v1/protocol.json \
  --output /absolute/local/new-prefix-study
```

ولإعادة السؤال نفسه مرتين على نسخة 70B المضغوطة الموجودة محليًا:

```sh
python tools/run_large_model_cache_pair.py \
  --runtime /absolute/local/llama-completion \
  --model /absolute/local/Llama-3.3-70B-Instruct-IQ2_XXS.gguf \
  --manifest models/llama70b.json \
  --output /absolute/local/new-consecutive-pair
```

استعمل `llama-completion` من الإصدار المثبت، لأن `llama-cli` رفض أحد
خيارات هذا البروتوكول قبل حساب النموذج. تتحقق الأداة من البصمة مرة
واحدة قبل التشغيل الأول، ثم تبدأ عملية مستقلة ثانية دون إعادة قراءة
الملف كاملًا أو مسح ذاكرة النظام. فحص هوية الملف بين العمليتين يقرأ
البيانات الوصفية فقط. لا تستعمل العملية الثانية KV من الأولى.

الأداة تقيس أول نص خرج على الشاشة، بما فيه وقت بدء البرنامج والحساب،
ولا تقيس لحظة أول token بدقة مكتبة داخلية. القراءة للتحقق قبل العملية
الأولى قد تدفئ ذاكرة الملفات، لذلك ليست هذه مقارنة قرص بارد وساخن.
تبقى حواجز الضغط والإيقاف فعالة، وتبقى الملفات والنتائج المتوقفة محفوظة.
