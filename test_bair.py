import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import progressbar
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from fvd.score import fvd as fvd_score
from lpips_metric.loss import PerceptualLoss


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument(
    '--data_root', default='/backup1/zhengchang/datasets/bair/', help='root directory for data')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--lpips_dir', default='lpips_metric/',
                    help='path of lpips')
parser.add_argument('--log_dir', default='',
                    help='directory to save generations to')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=2,
                    help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=28,
                    help='number of frames to predict')
parser.add_argument('--num_threads', type=int, default=0,
                    help='number of data loading threads')
parser.add_argument('--nsample', type=int, default=2, help='number of samples')
parser.add_argument('--rand_num', type=int, default=2,
                    help='number of samples')
parser.add_argument('--N', type=int, default=256, help='number of samples')


opt = parser.parse_args()
os.makedirs('%s' % opt.log_dir, exist_ok=True)


opt.n_eval = opt.n_past+opt.n_future
opt.max_step = opt.n_eval

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor


# ---------------- load the models  ----------------
lpips_model = PerceptualLoss(opt.lpips_dir)
tmp = torch.load(opt.model_path)
frame_predictor = tmp['frame_predictor']
posterior = tmp['posterior']
frame_predictor.eval()
posterior.eval()
encoder = tmp['encoder']
decoder = tmp['decoder']
encoder.eval()
decoder.eval()
frame_predictor.batch_size = opt.batch_size
posterior.batch_size = opt.batch_size
opt.g_dim = tmp['opt'].g_dim
opt.z_dim = tmp['opt'].z_dim
opt.num_digits = tmp['opt'].num_digits

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
encoder.cuda()
decoder.cuda()

# ---------------- set the options ----------------
opt.dataset = tmp['opt'].dataset
opt.last_frame_skip = tmp['opt'].last_frame_skip
opt.channels = tmp['opt'].channels
opt.image_width = tmp['opt'].image_width

print(opt)


# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.num_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.num_threads,
                         batch_size=opt.batch_size,
                         shuffle=False,
                         drop_last=True,
                         pin_memory=True)


def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch


training_batch_generator = get_training_batch()


def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch


testing_batch_generator = get_testing_batch()

# --------- eval funtions ------------------------------------


def make_gifs(x, idx, name):
    # get approx posterior sample
    frame_predictor.t_att, frame_predictor.s_att = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]
    all_gt = []
    all_gt.append(x[0])

    for i in range(1, opt.n_eval):
        all_gt.append(x[i])
        h = encoder(x_in)
        h_target = encoder(x[i])[0].detach()
        if opt.last_frame_skip or i < opt.n_past:
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        _, z_t, _ = posterior(h_target)  # take the mean
        if i < opt.n_past:
            frame_predictor(torch.cat([h, z_t], 1))
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1)).detach()
            x_in = decoder([h_pred, skip]).detach()
            posterior_gen.append(x_in)
    all_gt = torch.stack(all_gt, dim=0)

    nsample = opt.nsample
    ssim = np.zeros((opt.batch_size, nsample, opt.n_future))
    psnr = np.zeros((opt.batch_size, nsample, opt.n_future))
    lpips = np.zeros((opt.batch_size, nsample, opt.n_future))
    progress = progressbar.ProgressBar(maxval=nsample).start()
    all_gen = []
    for s in range(nsample):
        progress.update(s+1)
        gen_seq = []
        gt_seq = []
        frame_predictor.t_att, frame_predictor.s_att = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[i])[0].detach()
                _, z_t, _ = posterior(h_target)
            else:
                z_t = torch.cuda.FloatTensor(
                    opt.batch_size, opt.z_dim).normal_()
            if i < opt.n_past:
                frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
                all_gen[s].append(x_in)
            else:
                h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq.append(x_in.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())
                all_gen[s].append(x_in)
        lpips[:, s, :], ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(
            gt_seq, gen_seq, lpips_model)

    progress.finish()
    utils.clear_progressbar()

    ###### ssim ######
    best_seq_batch_ssim = []
    best_seq_batch_psnr = []
    best_seq_batch_lpips = []
    worst_seq_batch_ssim = []
    worst_seq_batch_psnr = []
    worst_seq_batch_lpips = []
    rand_seq_batch = []

    ssim_res = []
    psnr_res = []
    lpips_res = []

    ssim_worst_res = []
    psnr_worst_res = []
    lpips_worst_res = []

    for i in range(opt.batch_size):
        cur_best_seq_ssim = []
        cur_best_seq_psnr = []
        cur_best_seq_lpips = []
        cur_worst_seq_ssim = []
        cur_worst_seq_psnr = []
        cur_worst_seq_lpips = []
        cur_rand_seq = []

        # gifs = [ [] for t in range(opt.n_eval) ]
        # text = [ [] for t in range(opt.n_eval) ]

        mean_ssim = ssim[i]
        mean_psnr = psnr[i]
        mean_lpips = lpips[i]

        ordered_ssim = np.argsort(np.mean(mean_ssim, 1))
        ordered_psnr = np.argsort(np.mean(mean_psnr, 1))
        ordered_lpips = np.argsort(np.mean(mean_lpips, 1))

        best_idx_ssim = ordered_ssim[-1]
        best_idx_psnr = ordered_psnr[-1]
        best_idx_lpips = ordered_lpips[0]
        worst_idx_ssim = ordered_ssim[0]
        worst_idx_psnr = ordered_psnr[0]
        worst_idx_lpips = ordered_lpips[-1]

        # print(mean_ssim.shape, mean_ssim[ordered])
        ssim_res.append(mean_ssim[best_idx_ssim])
        psnr_res.append(mean_psnr[best_idx_psnr])
        lpips_res.append(mean_lpips[best_idx_lpips])
        ssim_worst_res.append(mean_ssim[worst_idx_ssim])
        psnr_worst_res.append(mean_psnr[worst_idx_psnr])
        lpips_worst_res.append(mean_lpips[worst_idx_lpips])

        for t in range(len(all_gen[0])):
            cur_best_seq_ssim.append(all_gen[best_idx_ssim][t][i])
            cur_best_seq_psnr.append(all_gen[best_idx_psnr][t][i])
            cur_best_seq_lpips.append(all_gen[best_idx_lpips][t][i])
            cur_worst_seq_ssim.append(all_gen[worst_idx_ssim][t][i])
            cur_worst_seq_psnr.append(all_gen[worst_idx_psnr][t][i])
            cur_worst_seq_lpips.append(all_gen[worst_idx_lpips][t][i])

            # 在五个sample上面算fvd
            cur_rand_seq_sample = []
            for rand_idx in range(opt.rand_num):
                cur_rand_seq_sample.append(all_gen[rand_idx][t][i])
            cur_rand_seq_sample = torch.stack(cur_rand_seq_sample)
            cur_rand_seq.append(cur_rand_seq_sample)

        # [t,c,h,w]
        cur_best_seq_ssim = torch.stack(cur_best_seq_ssim, dim=0)
        cur_best_seq_psnr = torch.stack(cur_best_seq_psnr, dim=0)
        cur_best_seq_lpips = torch.stack(cur_best_seq_lpips, dim=0)
        cur_worst_seq_ssim = torch.stack(cur_worst_seq_ssim, dim=0)
        cur_worst_seq_psnr = torch.stack(cur_worst_seq_psnr, dim=0)
        cur_worst_seq_lpips = torch.stack(cur_worst_seq_lpips, dim=0)
        cur_rand_seq = torch.stack(cur_rand_seq, dim=0)
        # print(cur_rand_seq.shape)
        best_seq_batch_ssim.append(cur_best_seq_ssim)
        best_seq_batch_psnr.append(cur_best_seq_psnr)
        best_seq_batch_lpips.append(cur_best_seq_lpips)
        worst_seq_batch_ssim.append(cur_worst_seq_ssim)
        worst_seq_batch_psnr.append(cur_worst_seq_psnr)
        worst_seq_batch_lpips.append(cur_worst_seq_lpips)
        rand_seq_batch.append(cur_rand_seq)
    # t,b
    ssim_res = np.stack(ssim_res, axis=1)
    psnr_res = np.stack(psnr_res, axis=1)
    lpips_res = np.stack(lpips_res, axis=1)
    ssim_worst_res = np.stack(ssim_worst_res, axis=1)
    psnr_worst_res = np.stack(psnr_worst_res, axis=1)
    lpips_worst_res = np.stack(lpips_worst_res, axis=1)
    # [t,b,c,h,w]
    best_seq_batch_ssim = torch.stack(best_seq_batch_ssim, dim=1)
    best_seq_batch_psnr = torch.stack(best_seq_batch_psnr, dim=1)
    best_seq_batch_lpips = torch.stack(best_seq_batch_lpips, dim=1)
    worst_seq_batch_ssim = torch.stack(worst_seq_batch_ssim, dim=1)
    worst_seq_batch_psnr = torch.stack(worst_seq_batch_psnr, dim=1)
    worst_seq_batch_lpips = torch.stack(worst_seq_batch_lpips, dim=1)
    rand_seq_batch = torch.cat(rand_seq_batch, dim=1)
    # print(rand_seq_batch.shape, best_seq_batch_psnr.shape)
    # print(all_gt.min(),all_gt.max())
    # print(best_seq_batch.min(),best_seq_batch.max())
    # print('ssim:',ssim_res)
    # fvd_ssim = fvd_score(all_gt.cpu(), best_seq_batch_ssim.cpu())
    # fvd_psnr = fvd_score(all_gt.cpu(), best_seq_batch_psnr.cpu())
    # print('fvd:',fvd_score_val)
    # print('gen:',best_seq_batch.shape)
    # print(cur_best_seq.shape)

    # rand_sidx = [np.random.randint(nsample) for s in range(3)]
    # for t in range(opt.n_eval):
    #     # gt
    #     gifs[t].append(add_border(x[t][i], 'green'))
    #     text[t].append('Ground\ntruth')
    #     #posterior
    #     if t < opt.n_past:
    #         color = 'green'
    #     else:
    #         color = 'red'
    #     gifs[t].append(add_border(posterior_gen[t][i], color))
    #     text[t].append('Approx.\nposterior')
    #     # best
    #     if t < opt.n_past:
    #         color = 'green'
    #     else:
    #         color = 'red'
    #     sidx = ordered[-1]
    #     gifs[t].append(add_border(all_gen[sidx][t][i], color))
    #     text[t].append('Best SSIM')
    #     # random 3
    #     for s in range(len(rand_sidx)):
    #         gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
    #         text[t].append('Random\nsample %d' % (s+1))

    # fname = '%s/%s_%d.gif' % (opt.log_dir, name, idx+i)
    # utils.save_gif_with_text(fname, gifs, text)
    return psnr_res, ssim_res, lpips_res, psnr_worst_res, ssim_worst_res, lpips_worst_res, all_gt.cpu(), best_seq_batch_ssim.cpu(), best_seq_batch_psnr.cpu(), best_seq_batch_lpips.cpu(), worst_seq_batch_ssim.cpu(), worst_seq_batch_psnr.cpu(), worst_seq_batch_lpips.cpu(), rand_seq_batch.cpu()


def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] = 0.7
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px


i = 0
dataset_len = len(test_loader)
# print(dataset_len)
# fvd_ssim = 0
# fvd_psnr = 0
ssim_best = []
psnr_best = []
lpips_best = []
ssim_worst = []
psnr_worst = []
lpips_worst = []

gt_seqs = []
gen_ssim_best_seqs = []
gen_psnr_best_seqs = []
gen_lpips_best_seqs = []
gen_psnr_worst_seqs = []
gen_ssim_worst_seqs = []
gen_lpips_worst_seqs = []
gen_rand_seqs = []
gt_rand_seqs = []
for test_x in test_loader:
    print(str(i+1)+'/'+str(dataset_len))
    # if(i>0):
    #     break
    test_x = utils.normalize_data(opt, dtype, test_x)
    # plot train
    # train_x = next(training_batch_generator)
    # make_gifs(train_x, i, 'train')

    # plot test
    # test_x = next(testing_batch_generator)
    cur_best_psnr, cur_best_ssim, cur_best_lpips, cur_worst_psnr, cur_worst_ssim, cur_worst_lpips, gt, gen_ssim_best, gen_psnr_best, gen_lpips_best, gen_ssim_worst, gen_psnr_worst, gen_lpips_worst, gen_rand = make_gifs(
        test_x, i, 'test')
    gt_seqs.append(gt)
    gen_ssim_best_seqs.append(gen_ssim_best)
    gen_psnr_best_seqs.append(gen_psnr_best)
    gen_lpips_best_seqs.append(gen_lpips_best)
    gen_ssim_worst_seqs.append(gen_ssim_worst)
    gen_psnr_worst_seqs.append(gen_psnr_worst)
    gen_lpips_worst_seqs.append(gen_lpips_worst)
    gen_rand_seqs.append(gen_rand)

    gt_rand = []
    for sample_idx in range(opt.rand_num):
        gt_rand.append(gt)
    gt_rand = torch.cat(gt_rand, dim=1)
    gt_rand_seqs.append(gt_rand)
    # print('FVD:', cur_fvd_ssim, cur_fvd_psnr, cur_psnr, cur_ssim,)
    psnr_best.append(cur_best_psnr)
    ssim_best.append(cur_best_ssim)
    lpips_best.append(cur_best_lpips)
    psnr_worst.append(cur_worst_psnr)
    ssim_worst.append(cur_worst_ssim)
    lpips_worst.append(cur_worst_lpips)
    # fvd_ssim += cur_fvd_ssim
    # fvd_psnr += cur_fvd_psnr
    i += 1
gt_seqs = torch.cat(gt_seqs, dim=1)
gen_ssim_best_seqs = torch.cat(gen_ssim_best_seqs, dim=1)
gen_psnr_best_seqs = torch.cat(gen_psnr_best_seqs, dim=1)
gen_lpips_best_seqs = torch.cat(gen_lpips_best_seqs, dim=1)
gen_ssim_worst_seqs = torch.cat(gen_ssim_worst_seqs, dim=1)
gen_psnr_worst_seqs = torch.cat(gen_psnr_worst_seqs, dim=1)
gen_lpips_worst_seqs = torch.cat(gen_lpips_worst_seqs, dim=1)

seqs_save = []
seqs_save.append(gen_ssim_best_seqs)
seqs_save.append(gen_psnr_best_seqs)
seqs_save.append(gen_lpips_best_seqs)
seqs_save.append(gen_ssim_worst_seqs)
seqs_save.append(gen_psnr_worst_seqs)
seqs_save.append(gen_lpips_worst_seqs)

gen_rand_seqs = torch.cat(gen_rand_seqs, dim=1)
gt_rand_seqs = torch.cat(gt_rand_seqs, dim=1)
# print(gt_seqs.shape)
# fvd_ssim = fvd_score(gt_seqs, gen_ssim_seqs)
# fvd_psnr = fvd_score(gt_seqs, gen_psnr_seqs)
# print(gt_rand_seqs.shape, gen_rand_seqs.shape)

psnr_best = np.concatenate(psnr_best, axis=1)
ssim_best = np.concatenate(ssim_best, axis=1)
lpips_best = np.concatenate(lpips_best, axis=1)
psnr_worst = np.concatenate(psnr_worst, axis=1)
ssim_worst = np.concatenate(ssim_worst, axis=1)
lpips_worst = np.concatenate(lpips_worst, axis=1)


# print(psnr_best.shape)
results_best = {}
results_best['psnr'] = psnr_best
results_best['ssim'] = ssim_best
results_best['lpips'] = lpips_best
results_worst = {}
results_worst['psnr'] = psnr_worst
results_worst['ssim'] = ssim_worst
results_worst['lpips'] = lpips_worst
np.savez_compressed(opt.log_dir + 'results_best.npz', **results_best)
np.savez_compressed(opt.log_dir + 'results_worst.npz', **results_worst)

levels = ['best', 'worst']
metrics = ['ssim', 'psnr', 'lpips']
save_idx = 0
for level in levels:
    for metric in metrics:
        # print(seqs_save[save_idx].min(), seqs_save[save_idx].max())
        cur_data = (seqs_save[save_idx].permute(
            1, 0, 3, 4, 2)*255).detach().numpy().astype(np.uint8)
        np.savez_compressed(opt.log_dir + metric + '_' +
                            level + '.npz', samples=cur_data[:, 2:, :])
        save_idx += 1
np.savez_compressed(opt.log_dir + 'gt.npz', (gt_seqs.permute(1,
                    0, 3, 4, 2)*255).detach().numpy().astype(np.uint8))
# print((gt_seqs.permute(1,0,3,4,2)*255).detach().numpy().astype(np.uint8).shape)
with open('best_model.txt', 'a+') as f:
    f.writelines(str(psnr_best.mean())+' '+str(ssim_best.mean()) +
                 ' ' + str(lpips_best.mean())+'\n')
print('Best scores:', psnr_best.mean(), ssim_best.mean(), lpips_best.mean())
print('Worst scores:', psnr_worst.mean(),
      ssim_worst.mean(), lpips_worst.mean())
fvd_rand = fvd_score(gt_rand_seqs, gen_rand_seqs)
print('FVD score:', fvd_rand)
with open('best_model.txt', 'a+') as f:
    f.writelines(str(psnr_best.mean())+' '+str(ssim_best.mean()) +
                 ' '+str(lpips_best.mean())+' '+str(fvd_rand) + '\n')
