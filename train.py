import argparse
import os
import time
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import SequencePairDataset
from model.encoder_decoder import EncoderDecoder
from evaluate import evaluate, get_bleu
from utils import to_np, trim_seqs

from tensorboardX import SummaryWriter
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu

from torch.optim.lr_scheduler import StepLR


def train(encoder_decoder: EncoderDecoder,
          train_data_loader: DataLoader,
          model_name,
          val_data_loader: DataLoader,
          keep_prob,
          teacher_forcing_schedule,
          lr,
          max_length,
          use_decay,
          data_path):

    global_step = 0
    loss_function = torch.nn.NLLLoss(ignore_index=0)
    optimizer = optim.Adam(encoder_decoder.parameters(), lr=lr)
    model_path = './saved/' + model_name + '/'

    if(use_decay == False):
        gamma = 1.0
    else:
        gamma = 0.5
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    #val_loss, val_bleu_score = evaluate(encoder_decoder, val_data_loader)

    best_bleu = 0.0

    for epoch, teacher_forcing in enumerate(teacher_forcing_schedule):
        #scheduler.step()
        print('epoch %i' % (epoch), flush=True)
        print('lr: ' + str(scheduler.get_lr()))

        for batch_idx, (input_idxs, target_idxs, input_tokens, target_tokens) in enumerate(tqdm(train_data_loader)):
            # input_idxs and target_idxs have dim (batch_size x max_len)
            # they are NOT sorted by length
            '''
            print(input_idxs[0])
            print(input_tokens[0])
            print(target_idxs[0])
            print(target_tokens[0])
            '''
            lengths = (input_idxs != 0).long().sum(dim=1)
            sorted_lengths, order = torch.sort(lengths, descending=True)

            input_variable = Variable(input_idxs[order, :][:, :max(lengths)])
            target_variable = Variable(target_idxs[order, :])

            optimizer.zero_grad()
            output_log_probs, output_seqs = encoder_decoder(input_variable,
                                                            list(sorted_lengths),
                                                            targets=target_variable,
                                                            keep_prob=keep_prob,
                                                            teacher_forcing=teacher_forcing)
            batch_size = input_variable.shape[0]

            flattened_outputs = output_log_probs.view(batch_size * max_length, -1)

            batch_loss = loss_function(flattened_outputs, target_variable.contiguous().view(-1))

            batch_loss.backward()
            optimizer.step()
            batch_outputs = trim_seqs(output_seqs)

            batch_targets = [[list(seq[seq > 0])] for seq in list(to_np(target_variable))]

            #batch_bleu_score = corpus_bleu(batch_targets, batch_outputs, smoothing_function=SmoothingFunction().method2)
            batch_bleu_score = corpus_bleu(batch_targets, batch_outputs)

            '''
            if global_step < 10 or (global_step % 10 == 0 and global_step < 100) or (global_step % 100 == 0 and epoch < 2):
                input_string = "Amy, Please schedule a meeting with Marcos on Tuesday April 3rd. Adam Kleczewski"
                output_string = encoder_decoder.get_response(input_string)
                writer.add_text('schedule', output_string, global_step=global_step)

                input_string = "Amy, Please cancel this meeting. Adam Kleczewski"
                output_string = encoder_decoder.get_response(input_string)
                writer.add_text('cancel', output_string, global_step=global_step)
            '''

            if global_step % 100 == 0:

                writer.add_scalar('train_batch_loss', batch_loss, global_step)
                writer.add_scalar('train_batch_bleu_score', batch_bleu_score, global_step)

                for tag, value in encoder_decoder.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value, global_step, bins='doane')
                    writer.add_histogram('grads/' + tag, to_np(value.grad), global_step, bins='doane')

            global_step += 1

            debug = False

            if(debug):
                if batch_idx == 5:
                    break

        val_loss, val_bleu_score = evaluate(encoder_decoder, val_data_loader)

        writer.add_scalar('val_loss', val_loss, global_step=global_step)
        writer.add_scalar('val_bleu_score', val_bleu_score, global_step=global_step)

        encoder_embeddings = encoder_decoder.encoder.embedding.weight.data
        encoder_vocab = encoder_decoder.lang.tok_to_idx.keys()
        writer.add_embedding(encoder_embeddings, metadata=encoder_vocab, global_step=0, tag='encoder_embeddings')

        decoder_embeddings = encoder_decoder.decoder.embedding.weight.data
        decoder_vocab = encoder_decoder.lang.tok_to_idx.keys()
        writer.add_embedding(decoder_embeddings, metadata=decoder_vocab, global_step=0, tag='decoder_embeddings')

        '''
        input_string = "Amy, Please schedule a meeting with Marcos on Tuesday April 3rd. Adam Kleczewski"
        output_string = encoder_decoder.get_response(input_string)
        writer.add_text('schedule', output_string, global_step=global_step)

        input_string = "Amy, Please cancel this meeting. Adam Kleczewski"
        output_string = encoder_decoder.get_response(input_string)
        writer.add_text('cancel', output_string, global_step=global_step)
        '''

        calc_bleu_score = get_bleu(encoder_decoder, data_path, None, 'dev')
        print('val loss: %.5f, val BLEU score: %.5f' % (val_loss, calc_bleu_score), flush=True)
        if(calc_bleu_score > best_bleu):
            print("Best BLEU score! Saving model...")
            best_bleu = calc_bleu_score
            torch.save(encoder_decoder, "%s%s_%i_%.3f.pt" % (model_path, model_name, epoch, calc_bleu_score))

        print('-' * 100, flush=True)

        scheduler.step()


def main(model_name, use_cuda, batch_size, teacher_forcing_schedule, keep_prob, val_size, lr, decoder_type, vocab_limit, hidden_size, embedding_size, max_length, data_path, use_decay, saved_model_path, seed=42):

    model_path = './saved/' + model_name + '/'

    # TODO: Change logging to reflect loaded parameters

    print("training %s with use_cuda=%s, batch_size=%i"% (model_name, use_cuda, batch_size), flush=True)
    print("teacher_forcing_schedule=", teacher_forcing_schedule, flush=True)
    print("keep_prob=%f, val_size=%f, lr=%f, decoder_type=%s, vocab_limit=%i, hidden_size=%i, embedding_size=%i, max_length=%i, seed=%i" % (keep_prob, val_size, lr, decoder_type, vocab_limit, hidden_size, embedding_size, max_length, seed), flush=True)

    if os.path.isdir(model_path):

        print("loading encoder and decoder from model_path", flush=True)
        #encoder_decoder = torch.load(model_path + model_name + '.pt')
        encoder_decoder = torch.load(saved_model_path)

        print("creating training and validation datasets with saved languages", flush=True)
        train_dataset = SequencePairDataset(data_path=data_path,
                                            maxlen=max_length,
                                            lang=encoder_decoder.lang,
                                            use_cuda=use_cuda,
                                            val_size=val_size,
                                            use_extended_vocab=(encoder_decoder.decoder_type=='copy'),
                                            data_type='train')

        val_dataset = SequencePairDataset(data_path=data_path,
                                          maxlen=max_length,
                                          lang=encoder_decoder.lang,
                                          use_cuda=use_cuda,
                                          val_size=val_size,
                                          use_extended_vocab=(encoder_decoder.decoder_type=='copy'),
                                          data_type='dev')

    else:
        os.mkdir(model_path)

        print("creating training and validation datasets", flush=True)
        train_dataset = SequencePairDataset(data_path=data_path,
                                            maxlen=max_length,
                                            vocab_limit=vocab_limit,
                                            use_cuda=use_cuda,
                                            val_size=val_size,
                                            seed=seed,
                                            use_extended_vocab=(decoder_type=='copy'),
                                            data_type='train')

        val_dataset = SequencePairDataset(data_path=data_path,
                                          maxlen=max_length,
                                          lang=train_dataset.lang,
                                          use_cuda=use_cuda,
                                          val_size=val_size,
                                          seed=seed,
                                          use_extended_vocab=(decoder_type=='copy'),
                                          data_type='dev')

        print("creating encoder-decoder model", flush=True)
        encoder_decoder = EncoderDecoder(train_dataset.lang,
                                         max_length,
                                         hidden_size,
                                         embedding_size,
                                         decoder_type)

        torch.save(encoder_decoder, model_path + '/%s.pt' % model_name)

    if use_cuda:
        encoder_decoder = encoder_decoder.cuda()
    else:
        encoder_decoder = encoder_decoder.cpu()
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)


    train(encoder_decoder,
          train_data_loader,
          model_name,
          val_data_loader,
          keep_prob,
          teacher_forcing_schedule,
          lr,
          encoder_decoder.decoder.max_length,
          use_decay,
          data_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse training parameters')
    parser.add_argument('model_name', type=str,
                        help='the name of a subdirectory of ./model/ that '
                             'contains encoder and decoder model files')

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--lr_decay', default='True', action='store_false')
    parser.add_argument('--saved_model_path', type=str)

    parser.add_argument('--epochs', type=int, default=50,
                        help='the number of epochs to train')

    parser.add_argument('--use_cuda', action='store_true',
                        help='flag indicating that cuda will be used')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='number of examples in a batch')

    parser.add_argument('--teacher_forcing_fraction', type=float, default=1,
                        help='fraction of batches that will use teacher forcing during training')

    parser.add_argument('--scheduled_teacher_forcing', action='store_true',
                        help='Linearly decrease the teacher forcing fraction '
                             'from 1.0 to 0.0 over the specified number of epocs')

    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='Probablity of keeping an element in the dropout step.')

    parser.add_argument('--val_size', type=float, default=0.1,
                        help='fraction of data to use for validation')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--decoder_type', type=str, default='copy',
                        help="Allowed values 'copy' or 'attn'")

    parser.add_argument('--vocab_limit', type=int, default=5000,
                        help='When creating a new Language object the vocab'
                             'will be truncated to the most frequently'
                             'occurring words in the training dataset.')

    parser.add_argument('--hidden_size', type=int, default=300,
                        help='The number of RNN units in the encoder. 2x this '
                             'number of RNN units will be used in the decoder')

    parser.add_argument('--embedding_size', type=int, default=300,
                        help='Embedding size used in both encoder and decoder')

    parser.add_argument('--max_length', type=int, default=50,
                        help='Sequences will be padded or truncated to this size.')

    args = parser.parse_args()

    writer = SummaryWriter('./logs/%s_%s' % (args.model_name, str(int(time.time()))))
    if args.scheduled_teacher_forcing:
        schedule = np.arange(1.0, 0.0, -1.0/args.epochs)
    else:
        schedule = np.ones(args.epochs) * args.teacher_forcing_fraction

    main(args.model_name, True, args.batch_size, schedule, args.keep_prob, args.val_size, args.lr, args.decoder_type, args.vocab_limit, args.hidden_size, args.embedding_size, args.max_length, args.data_path, args.lr_decay, args.saved_model_path)
    # main(str(int(time.time())), args.use_cuda, args.batch_size, schedule, args.keep_prob, args.val_size, args.lr, args.decoder_type, args.vocab_limit, args.hidden_size, args.embedding_size, args.max_length)
