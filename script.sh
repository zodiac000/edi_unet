#!/bin/zsh
name=10b
dirname=15b

for iter in {9..10}
do
    let previous=$iter-1
    let next=$iter+1

    #test all unlabeled images by current model
    sed -i -e 's/test_all = False/test_all = True/g' test_unet_cas_2.py
    sed -i -e s/$name\_$previous/$name\_$iter/g test_unet_cas_2.py
    if grep -q $name\_$iter test_unet_cas_2.py; then
        echo '==========================================================='
        echo '='$name\_$iter" found in test_unet_cas_2.py  -------  OK!"'='
        echo '==========================================================='
    else
        echo '==========================================================='
        echo '='$name\_$iter" not found in test_unet_cas_2.py  -------Failed!"
        echo '==========================================================='
        exit 0
    fi
    python test_unet_cas_2.py

    #test by crop discriminator
    sed -i -e s/$name\_$previous/$name\_$iter/g test_crop.py
    if grep -q $name\_$iter test_crop.py; then
        echo '==========================================================='
        echo '='$name\_$iter" found in test_crop.py  -------  OK!"
        echo '==========================================================='
    else
        echo '==========================================================='
        echo '='$name\_$iter" not found in test_crop.py  -------Failed!"
        echo '==========================================================='
        exit 0
    fi
    python test_crop.py

    #test by cutout discriminator
    sed -i -e s/$name\_$previous/$name\_$iter/g test_cutout.py
    if grep -q $name\_$iter test_cutout.py; then
        echo '==========================================================='
        echo '='$name\_$iter" found in test_cutout.py  -------  OK!"
        echo '==========================================================='
    else
        echo '==========================================================='
        echo '='$name\_$iter" not found in test_cutout.py  -------Failed!"
        echo '==========================================================='
        exit 0
    fi
    python test_cutout.py

    #filter qualified pseudo labels
    sed -i -e s/$name\_$previous/$name\_$iter/g filter.py
    if grep -q $name\_$iter filter.py; then
        echo '==========================================================='
        echo '='$name\_$iter" found in filter.py  -------  OK!"
        echo '==========================================================='
    else
        echo '==========================================================='
        echo '='$name\_$iter" not found in filter.py  -------Failed!"
        echo '==========================================================='
        exit 0
    fi
    python filter.py
    cat csv/$dirname/$name\_0.csv csv/eval/evaluation_all.csv > csv/$dirname/$name\_$next.csv

    #train the next iteration
    sed -i -e s/$name\_$iter/$name\_$next/g train_unet_cas_2.py
    if grep -q $name\_$next train_unet_cas_2.py; then
        echo '==========================================================='
        echo '='$name\_$next" found in train_unet_cas_2.py  -------  OK!"
        echo '==========================================================='
    else
        echo '==========================================================='
        echo '='$name\_$next" not found in train_unet_cas_2.py  -------Failed!"
        echo '==========================================================='
        exit 0
    fi
    python train_unet_cas_2.py

    #valid the next iteration
    sed -i -e s/$name\_$iter/$name\_$next/g test_unet_cas_2.py
    sed -i -e 's/test_all = True/test_all = False/g' test_unet_cas_2.py
    if grep -q $name\_$next test_unet_cas_2.py; then
        echo '==========================================================='
        echo '='$name\_$next" found in test_unet_cas_2.py  -------  OK!"
        echo '==========================================================='
    else
        echo '==========================================================='
        echo '='$name\_$next" not found in test_unet_cas_2.py  -------Failed!"
        echo '==========================================================='
        exit 0
    fi
    python test_unet_cas_2.py

    echo '==========================================================='
    echo '='Iteration $iter is Done!!!!!!!!!!!!!!!!!!!!!!!         '='
    echo '==========================================================='
done



