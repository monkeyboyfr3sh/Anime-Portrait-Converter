from __future__ import print_function
import argparse
import os
import random
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from models import Generator
from models import Discriminator

def image_to_vector(img, nz):
    preprocess = transforms.Compose([
        transforms.Grayscale(),
        transforms.CenterCrop(int(math.sqrt(nz)))
    ])
    vector = preprocess(img)
    vector = torch.flatten(vector,start_dim=1,end_dim=3)
    return vector[:,:,None,None]

if __name__ == "__main__":
    # Parse CMD line
    parser = argparse.ArgumentParser()
    parser.add_argument('--anime_dataroot', required=True, help='path to anime dataset')
    parser.add_argument('--human_dataroot', required=True, help='path to human dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

    opt = parser.parse_args()
    print(opt)
    
    #################################################
    # Parse the cmd inputs into vars
    #################################################
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    cudnn.benchmark = True
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    # make output dir if it doesn't exist
    if not (os.path.isdir(opt.outf)):
        try:
            os.makedirs(opt.outf)
        except:
            print('Could not make directory')
            exit(1)
    gen_path = os.path.join(opt.outf,'generator_chkpts')
    if not (os.path.isdir(gen_path)):
        try:
            os.mkdir(gen_path)
        except:
            print('Could not make directory')
            exit(1)
    disc_path = os.path.join(opt.outf,'discriminator_chkpts')
    if not (os.path.isdir(disc_path)):
        try:
            os.mkdir(disc_path)
        except:
            print('Could not make directory')
            exit(1)

    #################################################
    # Load human dataset/dataloader
    #################################################
    human_dataset = dset.ImageFolder(root=opt.human_dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    assert human_dataset
    human_dataloader = torch.utils.data.DataLoader(human_dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))
    human_iterator = iter(human_dataloader)
    fixed_human_sample = next(human_iterator)[0].to(device)
    #################################################
    # Load anime dataset/dataloader
    #################################################
    anime_dataset = dset.ImageFolder(root=opt.anime_dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    assert anime_dataset
    anime_dataloader = torch.utils.data.DataLoader(anime_dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))

    # One dataset will have different length from the other, so subset the longer one
    human_len = len(human_dataset)
    anime_len = len(anime_dataset)
    # Human list is bigger
    if(human_len>anime_len):
        print('Human dataset is larger than anime dataset, truncating human dataset')
        # TODO: Need to make this pull from random uinque indexs
        list_index = list(range(0,anime_len))
        human_dataset = torch.utils.data.Subset(human_dataset, list_index)
        assert human_dataset
        human_dataloader = torch.utils.data.DataLoader(human_dataset, batch_size=opt.batchSize,
                                                shuffle=True, num_workers=int(opt.workers))
        human_iterator = iter(human_dataloader)
        fixed_human_sample = next(human_iterator)[0].to(device)
    # Anime list is bigger
    elif(human_len<anime_len):
        print('Anime dataset is larger than human dataset, truncating anime dataset')
        list_index = list(range(0,human_len))
        anime_dataset = torch.utils.data.Subset(anime_dataset, list_index)
        assert anime_dataset
        anime_dataloader = torch.utils.data.DataLoader(anime_dataset, batch_size=opt.batchSize,
                                                shuffle=True, num_workers=int(opt.workers))
    nc=3
    #################################################
    # Make the models and load checkpoint if specified
    #################################################
    netG = Generator(ngpu,nz,nc,ngf).to(device)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = Discriminator(ngpu,nc,ndf).to(device)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)
    # Criterion function
    criterion = nn.BCELoss()
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    if opt.dry_run:
        opt.niter = 1
    real_label = 1
    human_fake_label = 0

    # Now do the training
    for epoch in range(opt.niter):
        # Update iterator with more data
        human_dataloader = torch.utils.data.DataLoader(human_dataset, batch_size=opt.batchSize,
                                                shuffle=True, num_workers=int(opt.workers))
        human_iterator = iter(human_dataloader)

        for i, data in enumerate(anime_dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,
                            dtype=real_cpu.dtype, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Get human sample
            human_sample = next(human_iterator)[0].to(device)
            human_fake = netG(image_to_vector(human_sample,nz))
            label.fill_(human_fake_label)
            output = netD(human_fake.detach())
            errD_human_fake = criterion(output, label)
            errD_human_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_human_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # human_fake labels are real for generator cost
            output = netD(human_fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, opt.niter, i, len(anime_dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                human_fake = netG(image_to_vector(fixed_human_sample,nz))
                vutils.save_image(human_fake.detach(),
                        '%s/human_fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)

            if opt.dry_run:
                break

        ############################
        # Now save checkpoints
        ###########################
        gen_fname = os.path.join(gen_path,'netG_epoch_%d.pth' % epoch)
        disc_fname = os.path.join(disc_path,'netD_epoch_%d.pth' % epoch)
        torch.save(netG.state_dict(), gen_fname)
        torch.save(netD.state_dict(), disc_fname)