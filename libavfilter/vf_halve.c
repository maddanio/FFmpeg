/*
 * Copyright (c) 2016 Floris Sluiter
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Pixel halve filter
 * This filter copies pixel by pixel a source frame to a target frame.
 * It halves the pixels to a new x,y destination based on two files ymap/xmap.
 * Map files are passed as a parameter and are in PGM format (P2 or P5),
 * where the values are y(rows)/x(cols) coordinates of the source_frame.
 * The *target* frame dimension is based on mapfile dimensions: specified in the
 * header of the mapfile and reflected in the number of datavalues.
 * Dimensions of ymap and xmap must be equal. Datavalues must be positive or zero.
 * Any datavalue in the ymap or xmap which value is higher
 * then the *source* frame height or width is silently ignored, leaving a
 * blank/chromakey pixel. This can safely be used as a feature to create overlays.
 *
 * Algorithm digest:
 * Target_frame[y][x] = Source_frame[ ymap[y][x] ][ [xmap[y][x] ];
 */

#include "libavutil/imgutils.h"
#include "libavutil/pixdesc.h"
#include "libavutil/opt.h"
#include "avfilter.h"
#include "formats.h"
#include "framesync.h"
#include "internal.h"
#include "video.h"

typedef struct HalveContext {
    const AVClass *class;
    int nb_planes;
    int nb_components;
    int step;
    FFFrameSync fs;

    void (*halve)(struct HalveContext *s, const AVFrame *in,
                  AVFrame *out);
} HalveContext;

#define OFFSET(x) offsetof(HalveContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption halve_options[] = {
    { NULL }
};

AVFILTER_DEFINE_CLASS(halve);

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUVA444P,
        AV_PIX_FMT_YUV444P,
        AV_PIX_FMT_YUVJ444P,
        AV_PIX_FMT_RGB24, AV_PIX_FMT_BGR24,
        AV_PIX_FMT_ARGB, AV_PIX_FMT_ABGR, AV_PIX_FMT_RGBA, AV_PIX_FMT_BGRA,
        AV_PIX_FMT_GBRP, AV_PIX_FMT_GBRAP,
        AV_PIX_FMT_YUV444P9, AV_PIX_FMT_YUV444P10, AV_PIX_FMT_YUV444P12,
        AV_PIX_FMT_YUV444P14, AV_PIX_FMT_YUV444P16,
        AV_PIX_FMT_YUVA444P9, AV_PIX_FMT_YUVA444P10, AV_PIX_FMT_YUVA444P16,
        AV_PIX_FMT_GBRP9, AV_PIX_FMT_GBRP10, AV_PIX_FMT_GBRP12,
        AV_PIX_FMT_GBRP14, AV_PIX_FMT_GBRP16,
        AV_PIX_FMT_GBRAP10, AV_PIX_FMT_GBRAP12, AV_PIX_FMT_GBRAP16,
        AV_PIX_FMT_RGB48, AV_PIX_FMT_BGR48,
        AV_PIX_FMT_RGBA64, AV_PIX_FMT_BGRA64,
        AV_PIX_FMT_NONE
    };
    AVFilterFormats *pix_formats = NULL;
    int ret;

    if (!(pix_formats = ff_make_format_list(pix_fmts))) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }
    if ((ret = ff_formats_ref(pix_formats, &ctx->inputs[0]->out_formats)) < 0 ||
        (ret = ff_formats_ref(pix_formats, &ctx->outputs[0]->in_formats)) < 0)
        goto fail;
    return 0;
fail:
    if (pix_formats)
        av_freep(&pix_formats->formats);
    av_freep(&pix_formats);
    return ret;
}

/**
 * halve_planar algorithm expects planes of same size
 * pixels are copied from source to target using :
 * Target_frame[y][x] = Source_frame[ ymap[y][x] ][ [xmap[y][x] ];
 */
static void halve_planar(HalveContext *s, const AVFrame *in,
                         AVFrame *out)
{
    int x , y, plane;

    for (plane = 0; plane < s->nb_planes ; plane++) {
        uint8_t *dst        = out->data[plane];
        const int dlinesize  = out->linesize[plane];
        const uint8_t *src  = in->data[plane];
        const int slinesize  = in->linesize[plane];
        for (y = 0; y < out->height; y++) {
            for (x = 0; x < out->width; x++, ++dst, src += 2) {
                dst[0] = (uint8_t)(
                    (
                        (uint16_t)(src[0]) +
                        (uint16_t)(src[slinesize]) +
                        (uint16_t)(src[1]) +
                        (uint16_t)(src[slinesize + 1])
                    ) >> 2);
            }
            src  += 2 * slinesize - in->width;
            dst  += dlinesize - out->width;
        }
    }
}

static void halve_planar16(HalveContext *s, const AVFrame *in,
                           AVFrame *out)
{
    int x , y, plane;

    for (plane = 0; plane < s->nb_planes ; plane++) {
        uint16_t *dst        = (uint16_t *)out->data[plane];
        const int dlinesize  = out->linesize[plane] / 2;
        const uint16_t *src  = (const uint16_t *)in->data[plane];
        const int slinesize  = in->linesize[plane] / 2;
        for (y = 0; y < out->height; y++) {
            for (x = 0; x < out->width; x++, ++dst, ++src) {
                dst[0] = (uint16_t)(
                    (
                        (uint32_t)(src[0]) +
                        (uint32_t)(src[slinesize]) +
                        (uint32_t)(src[1]) +
                        (uint32_t)(src[slinesize + 1])
                    ) >> 2);
            }
            src  += 2 * slinesize - in->width;
            dst  += dlinesize - out->width;
        }
    }
}



/**
 * halve_packed algorithm expects pixels with both padded bits (step) and
 * number of components correctly set.
 * pixels are copied from source to target using :
 * Target_frame[y][x] = Source_frame[ ymap[y][x] ][ [xmap[y][x] ];
 */

static void halve_packed(HalveContext *s, const AVFrame *in,
                         AVFrame *out)
{
    uint8_t *dst = out->data[0];
    const uint8_t *src  = in->data[0];
    const int dlinesize = out->linesize[0];
    const int slinesize = in->linesize[0];
    const int step = s->step;
    int c, x, y;

    for (y = 0; y < out->height; y++) {
        for (x = 0; x < out->width; x++, dst += step, src += step) {
            for (c = 0; c < s->nb_components; ++c) {
                dst[c] = (uint8_t)(
                    (
                        (uint16_t)(src[c]) +
                        (uint16_t)(src[slinesize + c]) +
                        (uint16_t)(src[step + c]) +
                        (uint16_t)(src[slinesize + step + c])
                    ) >> 2);
            }
        }
        src  += 2 * slinesize - in->width * step;
        dst  += dlinesize - out->width * step;
    }
}

static void halve_packed16(HalveContext *s, const AVFrame *in,
                           AVFrame *out)
{
    uint16_t *dst = (uint16_t *)out->data[0];
    const uint16_t *src  = (const uint16_t *)in->data[0];
    const int dlinesize = out->linesize[0] / 2;
    const int slinesize = in->linesize[0] / 2;
    const int step = s->step / 2;
    int c, x, y;

    for (y = 0; y < out->height; y++) {
        for (x = 0; x < out->width; x++, dst += step, src += step) {
            for (c = 0; c < s->nb_components; ++c) {
                dst[c] = (uint16_t)(
                    (
                        (uint32_t)(src[c]) +
                        (uint32_t)(src[slinesize + c]) +
                        (uint32_t)(src[step + c]) +
                        (uint32_t)(src[slinesize + step + c])
                    ) >> 2);
            }
        }
        src  += 2 * slinesize - in->width * step;
        dst  += dlinesize - out->width * step;
    }
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    HalveContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

    s->nb_planes = av_pix_fmt_count_planes(inlink->format);
    s->nb_components = desc->nb_components;

    if (desc->comp[0].depth == 8) {
        if (s->nb_planes > 1 || s->nb_components == 1) {
            s->halve = halve_planar;
        } else {
            s->halve = halve_packed;
        }
    } else {
        if (s->nb_planes > 1 || s->nb_components == 1) {
            s->halve = halve_planar16;
        } else {
            s->halve = halve_packed16;
        }
    }

    s->step = av_get_padded_bits_per_pixel(desc) >> 3;
    return 0;
}

static int process_frame(FFFrameSync *fs)
{
    AVFilterContext *ctx = fs->parent;
    HalveContext *s = fs->opaque;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *out, *in;
    int ret;

    if ((ret = ff_framesync_get_frame(&s->fs, 0, &in,   0)) < 0)
        return ret;

    if (ctx->is_disabled) {
        out = av_frame_clone(in);
        if (!out)
            return AVERROR(ENOMEM);
    } else {
        out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
        if (!out)
            return AVERROR(ENOMEM);
        av_frame_copy_props(out, in);

        s->halve(s, in, out);
    }
    out->pts = av_rescale_q(in->pts, s->fs.time_base, outlink->time_base);

    return ff_filter_frame(outlink, out);
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    HalveContext *s = ctx->priv;
    AVFilterLink *srclink = ctx->inputs[0];
    FFFrameSyncIn *in;
    int ret;

    outlink->w = srclink->w / 2;
    outlink->h = srclink->h / 2;
    outlink->time_base = srclink->time_base;
    outlink->sample_aspect_ratio = srclink->sample_aspect_ratio;
    outlink->frame_rate = srclink->frame_rate;

    ret = ff_framesync_init(&s->fs, ctx, 1);
    if (ret < 0)
        return ret;

    in = s->fs.in;
    in[0].time_base = srclink->time_base;
    in[0].sync   = 2;
    in[0].before = EXT_STOP;
    in[0].after  = EXT_STOP;
    s->fs.opaque   = s;
    s->fs.on_event = process_frame;

    return ff_framesync_configure(&s->fs);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *buf)
{
    HalveContext *s = inlink->dst->priv;
    return ff_framesync_filter_frame(&s->fs, inlink, buf);
}

static int request_frame(AVFilterLink *outlink)
{
    HalveContext *s = outlink->src->priv;
    return ff_framesync_request_frame(&s->fs, outlink);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    HalveContext *s = ctx->priv;

    ff_framesync_uninit(&s->fs);
}

static const AVFilterPad halve_inputs[] = {
    {
        .name         = "source",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
        .config_props = config_input,
    },
    { NULL }
};

static const AVFilterPad halve_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
        .request_frame = request_frame,
    },
    { NULL }
};

AVFilter ff_vf_halve = {
    .name          = "halve",
    .description   = NULL_IF_CONFIG_SMALL("Halve image"),
    .priv_size     = sizeof(HalveContext),
    .uninit        = uninit,
    .query_formats = query_formats,
    .inputs        = halve_inputs,
    .outputs       = halve_outputs,
    .priv_class    = &halve_class,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
};
