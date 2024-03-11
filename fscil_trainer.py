from copy import deepcopy

import torch.nn.functional as F
from tqdm import tqdm

from dataloader.data_utils import *
from models.Network import MYNET
from utils import *
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

class FSCILTrainer(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] * args.sessions
        self.model = MYNET(self.args)

        # self.micro_cluster=MicroCluster()
        if args.use_gpu:
            self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {'feature_extractor.' + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def get_optimizer_base(self):

        optimizer_resnet = torch.optim.SGD(self.model.feature_extractor.parameters(), lr=self.args.lr_base,
                                           momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler_resnet = torch.optim.lr_scheduler.StepLR(optimizer_resnet, step_size=self.args.step,
                                                               gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler_resnet = torch.optim.lr_scheduler.MultiStepLR(optimizer_resnet, milestones=self.args.milestones,
                                                                    gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler_resnet = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_resnet, T_max=self.args.epochs_base)

        return optimizer_resnet, scheduler_resnet

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)

            self.model = self.update_param(self.model, self.best_model_dict)

            optimizer, scheduler = self.get_optimizer_base()

            if session == 0:  # load base class train img label

                print('new classes for this session:\n', np.unique(train_set.targets))

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess

                    tl, ta = resnet_train(self.model, trainloader, optimizer, epoch, args, session)

                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args, session)

                    visualize(self.model, testloader, epoch, args, session)

                    # test model with all seen class
                    tsl, tsa = test(self.model, testloader, epoch, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(self.model.state_dict(), save_model_dir)
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    result_list.append(
                        'epoch:%03d,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))

                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))


            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args, session)

                visualize(self.model, testloader, epoch, args, session)

                tsl, tsa = test(self.model, testloader, 0, args, session)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):
        mode = 'baseline'
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)

        ## add the slurm process id
        job_id = os.environ.get('SLURM_JOB_ID')
        self.args.save_path = self.args.save_path + 'slurm_id_%s/' % str(job_id)

        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None


def resnet_train(model, trainloader, optimizer, epoch, args, session):
    tl = Averager()
    ta = Averager()
    model = model.train()

    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        if args.use_gpu:
            data = batch[0].cuda()
            true_label = batch[1].long().cuda()
        else:
            data = batch[0]
            true_label = batch[1].long()

        logits = model(data)
        total_loss = F.cross_entropy(logits, true_label)
        acc = count_acc(logits, true_label)
        tqdm_gen.set_description(
            'Session 0, epo {}, total loss={:.4f} acc={:.4f}'.format(epoch, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def test(model, testloader, epoch, args, session):
    model = model.eval()
    vl = Averager()
    va = Averager()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            if args.use_gpu:
                data = batch[0].cuda()
                true_label = batch[1].long().cuda()
            else:
                data = batch[0]
                true_label = batch[1].long()

            logits = model(data)
            total_loss = F.cross_entropy(logits, true_label)
            if session == 0:
                acc = count_acc(logits, true_label)
            else:
                acc = count_base_session_acc(logits, true_label)
            tqdm_gen.set_description('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, total_loss, acc))
            vl.add(total_loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va


def replace_base_fc(trainset, transform, model, args, session):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base,
                                              num_workers=args.num_workers, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]

            embedding = model.feature_extractor(data).squeeze()

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    if session == 0:

        for class_index in range(args.base_class + args.way * session):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)

        proto_list = torch.stack(proto_list, dim=0)
        model.feature_extractor.fc.weight.data[:args.base_class] = proto_list

    else:
        for class_index in range(args.base_class + args.way * (session - 1), args.base_class + args.way * session):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)

        proto_list = torch.stack(proto_list, dim=0)
        model.feature_extractor.fc.weight.data[
        args.base_class + args.way * (session - 1): args.base_class + args.way * session] = proto_list

    return model

def visualize(model, testloader, epoch, args, session):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    dataset_embeddings = []
    fake_embeddings = []
    true_labels = []
    fake_labels = []
    # data_list=[]
    # with torch.no_grad():
    for i, batch in enumerate(testloader):
        data, label = [_.cuda() for _ in batch]

        real_embedding = model.feature_extractor(data).squeeze()

        # Convert embeddings and labels to numpy arrays
        dataset_embeddings.append(real_embedding.detach().cpu().numpy())
        true_labels.append(label.cpu().numpy())

    dataset_embeddings = np.concatenate(dataset_embeddings, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    class_indices = np.where(true_labels >= 55)[0]

    dataset_embeddings = dataset_embeddings[class_indices]
    true_labels = true_labels[class_indices]

    prototypes = model.feature_extractor.fc.weight.data[55:(60+5*session)].clone().detach().cpu().numpy()

    combine_data=np.vstack([prototypes, dataset_embeddings])

    # Reduce the dimensionality of real embeddings using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    combine_embeddings_2d = tsne.fit_transform(combine_data)

    proto_embeddings_2d, real_embeddings_2d = combine_embeddings_2d[:(5+5*session)], combine_embeddings_2d[(5+5*session):]

    # Create a scatter plot of the t-SNE embeddings for real images
    plt.scatter(real_embeddings_2d[:, 0], real_embeddings_2d[:, 1], c=true_labels)
    plt.scatter(proto_embeddings_2d[:5, 0], proto_embeddings_2d[:5, 1], c='red',marker='o', s=50, label='Prototypes')
    if session==1:
        plt.scatter(proto_embeddings_2d[5:, 0], proto_embeddings_2d[5:, 1], c='red', marker='x', s=50, label='Prototypes')
    plt.title("t-SNE Visualization of Test Samples on Session {}".format(session))
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig("tsne{}.png".format(session))
    plt.show()

    print('aa')
