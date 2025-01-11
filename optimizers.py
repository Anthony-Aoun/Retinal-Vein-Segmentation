"""This file contains old - more or less naive - optimizers"""

def find_best_threshold():
    scores = []
    best_score = 0
    best_thres = 50
    for thres in range(50, 120, 2):
        print(thres)
        score = 0
        i = 1
        # for i in range(1, 10):
        img, img_mask, img_GT = load_image(i)
        img_out = img_segmentation(img, img_mask, thres)
        _, _, F1, _, _ = evaluate(img_out, img_GT)
        score += F1

        # score /= 10
        print(score)
        scores.append(score)

        if score > best_score:
            best_score = score
            best_thres = thres
    print('Best threshold: ', best_thres, ', F1: ', best_score)
    plt.plot(scores, list(range(50, 250, 4)))
    plt.show()
    return best_thres


def find_best_contrast():
    scores = []
    best_score = 0
    best_contrast = 1
    for contrast in range(50, 70):
        print(contrast)
        score = 0
        for i in range(1, 11):
            img, img_mask, img_GT = load_image(i)
            img_out = img_segmentation(
                img, img_mask, threshold-lower, threshold, contrast/100, False)
            _, _, F1, _, _ = evaluate(img_out, img_GT)
            score += F1
            print(F1)
        score /= 10
        print(score)
        if score > best_score:
            best_score = score
            best_contrast = contrast
    print('Best contrast: ', best_contrast/100, ', F1: ', best_score)
    return best_contrast


def find_best_threshold_couple():
    best_score = 0
    best_thres = 50
    best_lower = 0
    scores_thres = []
    for thres in range(40, 101, 2):  # range(40, 70, 2):
        print(thres)
        scores_lower = []
        for lower in range(0, min(thres, 21), 2):
            scores_contrast = []
            for contrast in range(3, 16, 1):
                score = 0
                # for i in range(1, 11):
                i = 1
                if True:
                    img, img_mask, img_GT = load_image(i)
                    img_out = img_segmentation(
                        img, img_mask, thres-lower, thres, round(contrast/5, 2))
                    try:
                        _, _, F1, _, _ = evaluate(img_out, img_GT)
                    except:
                        F1 = 0
                    score += F1
                #score /= 10

                print('Threshold: ', thres, ', Contrast: ',
                      round(contrast/5, 2), ', Lower: ',
                      lower, ', F1: ', score)
                if score > best_score:
                    best_score = score
                    best_thres = thres
                    best_lower = lower
                    best_contrast = contrast/5
                if len(scores_contrast) >= 3 and score < np.mean(scores_contrast[-3:]):
                    break
                else:
                    scores_contrast.append(score)
            if len(scores_lower) >= 3 and np.mean(scores_contrast) < np.mean(scores_lower[-3:]):
                break
            else:
                scores_lower.append(np.mean(scores_contrast))
        if len(scores_thres) >= 3 and np.max(scores_lower) < np.max(scores_thres) - 0.003:
            break
        else:
            scores_thres.append(np.max(scores_lower))
    print('Best threshold: ', best_thres, ', contrast: ',
          best_contrast, ', Lower: ',
          best_lower, ', F1: ', best_score)
    return best_thres, best_contrast, best_lower


