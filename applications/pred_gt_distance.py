import json
import math
from lib.utils.utils import AverageMeter
from lib.config import cfg

LIMBS = {
        'left_arm': [0, 2*3, 4*3],
        'right_arm': [1*3, 3*3, 5*3],
        'left_leg': [6*3, 8*3, 10*3],
        'right_leg': [7*3, 9*3, 11*3]
    }

left_arm = AverageMeter()
right_arm = AverageMeter()
left_leg = AverageMeter()
right_leg = AverageMeter()
total_err = AverageMeter()
def get_limb(the_limb, keypoints):
    limb = [
        [keypoints[LIMBS[the_limb][0]], keypoints[LIMBS[the_limb][0]+1]],
        [keypoints[LIMBS[the_limb][1]], keypoints[LIMBS[the_limb][1] + 1]],
        [keypoints[LIMBS[the_limb][2]], keypoints[LIMBS[the_limb][2] + 1]]
    ]
    return limb

def limb_distance(gt, pred, the_image):
    distance_sum = 0
    counter = 0
    for i in range(3):
        if gt[i] != [0, 0]:  # nn considero i giunti non visibili
            distance_sum += math.dist(gt[i], pred[i])
            counter += 1
    result = distance_sum/counter if (counter != 0) else 0
    # return math.sqrt((result**2)/(the_image['width']*the_image['height']))
    return 1 - (result / math.sqrt((the_image['width']**2) + (the_image['height']**2)))


def pred_gt_distance(predicted, ground_truth, logger):

    limbs = {
        'left_arm': [0, 2, 4],
        'right_arm': [1, 3, 5],
        'left_leg': [6, 8, 10],
        'right_leg': [7, 9, 11]
    }

    with open(predicted, "r") as fp:
        # Load the predictions dictionary from the file
        predictions = json.load(fp)
    with open(ground_truth, "r") as fp:
        # Load the ground truth dictionary from the file
        the_json = json.load(fp)
        gts = the_json['annotations']
        images = the_json['images']


    for prediction in predictions:
        # annootations dell'immagine in esame
        the_gt = list(filter(lambda gt: gt['image_id'] == prediction['image_id'], gts))[0]
        # caratteristiche dell'immagine in esame
        the_image = list(filter(lambda image: image['id'] == prediction['image_id'], images))[0]

        kpts = the_gt['keypoints']
        kpts_pred = prediction['keypoints']

        left_arm_gt = get_limb('left_arm', kpts)
        right_arm_gt = get_limb('right_arm', kpts)
        left_leg_gt = get_limb('left_leg', kpts)
        right_leg_gt = get_limb('right_leg', kpts)

        left_arm_pred = get_limb('left_arm', kpts_pred)
        right_arm_pred = get_limb('right_arm', kpts_pred)
        left_leg_pred = get_limb('left_leg', kpts_pred)
        right_leg_pred = get_limb('right_leg', kpts_pred)

        left_arm_err = limb_distance(left_arm_gt, left_arm_pred, the_image)
        right_arm_err = limb_distance(right_arm_gt, right_arm_pred, the_image)
        left_leg_err = limb_distance(left_leg_gt, left_leg_pred, the_image)
        right_leg_err = limb_distance(right_leg_gt, right_leg_pred, the_image)

        left_arm.update(left_arm_err)
        right_arm.update(right_arm_err)
        left_leg.update(left_leg_err)
        right_leg.update(right_leg_err)

        total_err.update(
            (left_arm_err + right_arm_err + left_leg_err + right_leg_err) / 4
        )


        print(f"Predizioni per immagine: {the_gt['image_id']}")
        print('left_arm  : {:.2e}'.format(left_arm_err))
        print('right_arm : {:.2e}'.format(right_arm_err))
        print('left_leg  : {:.2e}'.format(left_leg_err))
        print('right_leg : {:.2e}'.format(right_leg_err))

        print('-----------------------------------------------------------------')


    logger.info('****************************************************')
    logger.info("  Valori medi totali")
    logger.info('left_arm  : {:.2e}'.format(left_arm.avg))
    logger.info('right_arm : {:.2e}'.format(right_arm.avg))
    logger.info('left_leg  : {:.2e}'.format(left_leg.avg))
    logger.info('right_leg : {:.2e}'.format(right_leg.avg))
    logger.info('Total     : {:.2e}'.format(total_err.avg))

if __name__ == '__main__':
    pred_gt_distance(
        '../output/baby_pose_kpt/hrnet_dekr/w32_4x_reg03_bs10_512_adam_lr1e-3_babypose_x300/results/keypoints_testregression_results.json',
        '../data/babypose/json/babypose_test.json'
    )
